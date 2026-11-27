//! Where a shared entry lives: [`ShardRing`], rendezvous hashing over the pool's inference nodes.
//!
//! Every node that serves the shared cache ranks the nodes for a shard the same way: by a
//! SHA-256 score over the node id and the shard (a [`scope_key`](crate::scope_key) for
//! [`SemanticCache`](crate::SemanticCache) entries, a [`Key`](crate::Key) for exact ones). The
//! highest is the **home** node and the next the **replica** ([`ShardPlan`]). Nothing is
//! coordinated and nothing is stored: nodes that agree on the member list agree on every plan.
//!
//! Rendezvous hashing moves as little as possible when the pool changes. When a node leaves,
//! only the shards it held move, and its replica becomes their home (so they are still there).
//! When a node joins, it takes over only the shards it now ranks first for, and their old home
//! becomes their replica.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

/// The home node and replica for one shard, from [`ShardRing::plan`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ShardPlan {
    /// Ranked first: asked first, written first.
    pub home: String,
    /// Ranked second; `None` in a one-node pool.
    pub replica: Option<String>,
}

impl ShardPlan {
    pub fn new(home: impl Into<String>, replica: Option<String>) -> Self {
        Self { home: home.into(), replica }
    }

    /// The home node, then the replica if there is one.
    pub fn owners(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.home.as_str()).chain(self.replica.as_deref())
    }

    /// `node` is the home or the replica.
    pub fn owns(&self, node: &str) -> bool {
        self.owners().any(|o| o == node)
    }
}

/// The inference nodes that share a cache, and which of them holds each shard.
///
/// Node ids are opaque strings (for Weirdnet, the peer ids). Every node must use the same ids.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShardRing {
    nodes: BTreeSet<String>,
}

impl ShardRing {
    /// A ring over `nodes` (duplicates count once).
    pub fn new<I, S>(nodes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self { nodes: nodes.into_iter().map(Into::into).collect() }
    }

    /// Adds `node`; `false` when it was already a member.
    pub fn insert(&mut self, node: impl Into<String>) -> bool {
        self.nodes.insert(node.into())
    }

    /// Removes `node`; `false` when it wasn't a member.
    pub fn remove(&mut self, node: &str) -> bool {
        self.nodes.remove(node)
    }

    pub fn contains(&self, node: &str) -> bool {
        self.nodes.contains(node)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The members, sorted.
    pub fn nodes(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(String::as_str)
    }

    /// Every member, best first for `shard`: the order a lookup tries them in.
    pub fn ranked(&self, shard: impl AsRef<[u8]>) -> Vec<&str> {
        let shard = shard.as_ref();
        let mut scored: Vec<(u64, &str)> = self.nodes.iter().map(|n| (score(n, shard), n.as_str())).collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        scored.into_iter().map(|(_, n)| n).collect()
    }

    /// The home node and replica for `shard`, or `None` for an empty ring.
    pub fn plan(&self, shard: impl AsRef<[u8]>) -> Option<ShardPlan> {
        let mut ranked = self.ranked(shard).into_iter();
        let home = ranked.next()?.to_owned();
        Some(ShardPlan { home, replica: ranked.next().map(str::to_owned) })
    }
}

impl<S: Into<String>> FromIterator<S> for ShardRing {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        Self::new(iter)
    }
}

/// The rendezvous score of `node` for `shard`: the first 8 bytes (big-endian) of SHA-256 over a
/// domain tag and both, length-prefixed. Part of the shared-cache contract: changing it moves
/// every shard.
fn score(node: &str, shard: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(b"weirdnet-semcache/shard/v1");
    h.update((node.len() as u64).to_le_bytes());
    h.update(node.as_bytes());
    h.update((shard.len() as u64).to_le_bytes());
    h.update(shard);
    let d = h.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&d[..8]);
    u64::from_be_bytes(first)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const PINNED: u64 = 0x2887_d89b_7b9d_d087;
    const PINNED_HOME: &str = "node-b";
    const PINNED_REPLICA: &str = "node-c";

    fn nodes(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("node-{i}")).collect()
    }

    fn scopes(n: usize) -> Vec<String> {
        (0..n).map(|i| crate::scope_key(&["model", &i.to_string()])).collect()
    }

    #[test]
    fn plans_are_deterministic_and_distinct() {
        let a = ShardRing::new(nodes(5));
        let b: ShardRing = nodes(5).into_iter().rev().collect();
        for s in scopes(50) {
            let p = a.plan(&s).unwrap();
            assert_eq!(Some(&p), b.plan(&s).as_ref(), "member order doesn't matter");
            assert_ne!(Some(p.home.as_str()), p.replica.as_deref());
            assert!(p.owns(&p.home) && p.owns(p.replica.as_deref().unwrap()));
            assert_eq!(p.owners().count(), 2);
            assert_eq!(a.ranked(&s)[..2], [p.home.as_str(), p.replica.as_deref().unwrap()]);
        }
    }

    #[test]
    fn small_rings() {
        assert_eq!(ShardRing::default().plan("s"), None);
        let one = ShardRing::new(["only"]);
        assert_eq!(one.plan("s"), Some(ShardPlan::new("only", None)));
        let mut r = ShardRing::new(["a", "a", "b"]);
        assert_eq!(r.len(), 2);
        assert!(!r.insert("a") && r.insert("c") && r.remove("c") && !r.remove("c"));
        assert_eq!(r.nodes().collect::<Vec<_>>(), ["a", "b"]);
        assert!(r.contains("a") && !r.contains("c") && !r.is_empty());
    }

    #[test]
    fn the_score_is_pinned() {
        // Part of the contract between nodes (and versions): a change here moves every shard.
        assert_eq!(score("node-a", b"scope"), PINNED);
        let r = ShardRing::new(["node-a", "node-b", "node-c"]);
        assert_eq!(r.plan(b"scope").unwrap(), ShardPlan::new(PINNED_HOME, Some(PINNED_REPLICA.into())));
    }

    #[test]
    fn shards_spread_over_the_nodes() {
        let r = ShardRing::new(nodes(4));
        let mut homes: HashMap<String, usize> = HashMap::new();
        for s in scopes(4000) {
            *homes.entry(r.plan(&s).unwrap().home).or_default() += 1;
        }
        assert_eq!(homes.len(), 4);
        for (n, count) in homes {
            assert!((700..1300).contains(&count), "{n} is home for {count} of 4000");
        }
    }

    #[test]
    fn a_leaving_node_moves_only_its_shards_to_their_replica() {
        let before = ShardRing::new(nodes(5));
        let mut after = before.clone();
        after.remove("node-2");
        for s in scopes(500) {
            let (p, q) = (before.plan(&s).unwrap(), after.plan(&s).unwrap());
            if p.home == "node-2" {
                assert_eq!(Some(q.home.as_str()), p.replica.as_deref(), "the replica takes over");
            } else {
                assert_eq!(q.home, p.home, "other shards stay home");
            }
        }
    }

    #[test]
    fn a_joining_node_takes_shards_and_their_old_home_becomes_the_replica() {
        let before = ShardRing::new(nodes(4));
        let mut after = before.clone();
        after.insert("node-new");
        let mut moved = 0;
        for s in scopes(1000) {
            let (p, q) = (before.plan(&s).unwrap(), after.plan(&s).unwrap());
            if q.home == "node-new" {
                moved += 1;
                assert_eq!(q.replica.as_deref(), Some(p.home.as_str()), "the old home still has the entries");
            } else {
                assert_eq!(q.home, p.home);
            }
        }
        assert!((100..300).contains(&moved), "about a fifth move: {moved}");
    }
}
