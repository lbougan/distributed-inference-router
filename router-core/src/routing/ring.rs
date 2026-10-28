//! Consistent hashing with virtual nodes. The ring is built once from the full backend
//! list; health changes only affect which ring points are skipped at lookup time, so a
//! backend going down moves ~1/n of the keys instead of all of them.

use sha2::{Digest, Sha256};

use crate::backend::BackendId;

pub const DEFAULT_VNODES: usize = 128;

pub fn hash_key(key: &[u8]) -> u64 {
    let digest = Sha256::digest(key);
    u64::from_le_bytes(digest[..8].try_into().expect("digest has 32 bytes"))
}

#[derive(Debug, Clone)]
pub struct HashRing {
    points: Vec<(u64, BackendId)>,
}

impl HashRing {
    pub fn new(num_backends: usize, vnodes: usize) -> Self {
        let mut points = Vec::with_capacity(num_backends * vnodes);
        for id in 0..num_backends {
            for v in 0..vnodes {
                points.push((hash_key(format!("backend-{id}-vnode-{v}").as_bytes()), id));
            }
        }
        points.sort_unstable();
        Self { points }
    }

    /// First ring point at or after `key_hash` whose backend is routable, wrapping around.
    pub fn lookup(&self, key_hash: u64, routable: impl Fn(BackendId) -> bool) -> Option<BackendId> {
        if self.points.is_empty() {
            return None;
        }
        let n = self.points.len();
        let start = self.points.partition_point(|(h, _)| *h < key_hash);
        (0..n)
            .map(|i| self.points[(start + i) % n].1)
            .find(|id| routable(*id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(n: usize) -> Vec<u64> {
        (0..n)
            .map(|i| hash_key(format!("prompt-{i}").as_bytes()))
            .collect()
    }

    #[test]
    fn same_key_same_backend() {
        let ring = HashRing::new(5, DEFAULT_VNODES);
        let k = hash_key(b"hello");
        assert_eq!(ring.lookup(k, |_| true), ring.lookup(k, |_| true));
    }

    #[test]
    fn empty_ring_or_nothing_routable_gives_none() {
        assert!(HashRing::new(0, DEFAULT_VNODES)
            .lookup(1, |_| true)
            .is_none());
        assert!(HashRing::new(3, DEFAULT_VNODES)
            .lookup(1, |_| false)
            .is_none());
    }

    #[test]
    fn removing_one_of_ten_backends_moves_at_most_15_percent_of_keys() {
        let ring = HashRing::new(10, DEFAULT_VNODES);
        let keys = keys(10_000);
        let before: Vec<usize> = keys
            .iter()
            .map(|k| ring.lookup(*k, |_| true).unwrap())
            .collect();
        let after: Vec<usize> = keys
            .iter()
            .map(|k| ring.lookup(*k, |id| id != 3).unwrap())
            .collect();
        let moved = before.iter().zip(&after).filter(|(b, a)| b != a).count();
        assert!(moved <= 1_500, "moved {moved} of 10000");
        for (b, a) in before.iter().zip(&after) {
            if b != a {
                assert_eq!(*b, 3, "only keys from the removed backend may move");
            }
        }
        assert!(!after.contains(&3));
    }

    #[test]
    fn keys_are_spread_evenly() {
        let ring = HashRing::new(10, DEFAULT_VNODES);
        let mut counts = [0usize; 10];
        for k in keys(20_000) {
            counts[ring.lookup(k, |_| true).unwrap()] += 1;
        }
        for c in counts {
            assert!((1_000..=3_000).contains(&c), "uneven: {counts:?}");
        }
    }
}
