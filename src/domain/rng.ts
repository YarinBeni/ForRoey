/**
 * Small deterministic hashing + PRNG utilities.
 *
 * Slot times must be reproducible for the same (seed, date) so that the app
 * shows the same schedule after a restart and so notifications match what is
 * on screen. We therefore avoid Math.random() in the domain layer entirely.
 */

/**
 * FNV-1a 32-bit hash of a string. Returns an unsigned 32-bit integer.
 * See http://www.isthe.com/chongo/tech/comp/fnv/
 */
export function hashString(s: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < s.length; i++) {
    hash ^= s.charCodeAt(i);
    // hash *= 16777619 (FNV prime), done with shifts to stay in 32-bit math.
    hash = (hash + ((hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24))) >>> 0;
  }
  return hash >>> 0;
}

/**
 * mulberry32: a tiny, fast 32-bit PRNG with a decent distribution.
 * Returns a function that yields floats in [0, 1).
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
