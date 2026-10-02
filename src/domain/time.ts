/**
 * Time-zone helpers built only on `Intl` (no date libraries).
 *
 * The app's "day" is defined in the user's time zone, identified by a
 * `dateKey` of the form "YYYY-MM-DD". Instants are plain `Date` objects (UTC).
 */

const MINUTES_PER_DAY = 1440;
const MS_PER_MINUTE = 60_000;
const MS_PER_DAY = 86_400_000;

type WallClock = {
  year: number;
  month: number; // 1..12
  day: number; // 1..31
  hour: number; // 0..23
  minute: number; // 0..59
  second: number; // 0..59
};

const formatterCache = new Map<string, Intl.DateTimeFormat>();

/** Memoised formatter: creating Intl.DateTimeFormat is comparatively expensive. */
function formatterFor(tz: string): Intl.DateTimeFormat {
  let f = formatterCache.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    formatterCache.set(tz, f);
  }
  return f;
}

/** Break an instant into its wall-clock components in `tz`. */
export function wallClockInTz(date: Date, tz: string): WallClock {
  const parts = formatterFor(tz).formatToParts(date);
  const get = (type: Intl.DateTimeFormatPartTypes): number => {
    const p = parts.find((x) => x.type === type);
    return p ? parseInt(p.value, 10) : 0;
  };
  let hour = get('hour');
  // Some engines report midnight as "24" despite hourCycle h23; normalise.
  if (hour === 24) hour = 0;
  return {
    year: get('year'),
    month: get('month'),
    day: get('day'),
    hour,
    minute: get('minute'),
    second: get('second'),
  };
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

function toDateKey(year: number, month: number, day: number): string {
  return `${year}-${pad2(month)}-${pad2(day)}`;
}

/** Parse "YYYY-MM-DD" into numeric parts. Throws on malformed input. */
export function parseDateKey(dateKey: string): { year: number; month: number; day: number } {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateKey);
  if (!m) throw new Error(`Invalid dateKey "${dateKey}" (expected YYYY-MM-DD)`);
  return { year: Number(m[1]), month: Number(m[2]), day: Number(m[3]) };
}

/** "YYYY-MM-DD" of the given instant as seen in `tz`. */
export function dateKeyInTz(date: Date, tz: string): string {
  const w = wallClockInTz(date, tz);
  return toDateKey(w.year, w.month, w.day);
}

/** Minutes after local midnight (0..1439) of the given instant in `tz`. */
export function minuteOfDayInTz(date: Date, tz: string): number {
  const w = wallClockInTz(date, tz);
  return w.hour * 60 + w.minute;
}

/**
 * Offset of `tz` from UTC at the given instant, in minutes.
 * Positive east of Greenwich (Asia/Jerusalem → +120 in winter, +180 in summer).
 */
export function tzOffsetMinutes(date: Date, tz: string): number {
  const w = wallClockInTz(date, tz);
  const asIfUtc = Date.UTC(w.year, w.month - 1, w.day, w.hour, w.minute, w.second);
  // Drop sub-second noise so the offset is a whole number of minutes.
  const wholeSeconds = Math.floor(date.getTime() / 1000) * 1000;
  return Math.round((asIfUtc - wholeSeconds) / MS_PER_MINUTE);
}

/**
 * Convert a wall-clock time in `tz` to an instant.
 *
 * `minuteOfDay` may be >= 1440 (or negative); whole days are carried over into
 * the date. The algorithm guesses the UTC instant assuming the zone's offset at
 * the naive UTC time, then re-evaluates the offset at that guess. Two passes
 * are enough to land on the right side of a DST transition. Times that fall in
 * a DST gap resolve to the instant after the gap; ambiguous times (fall-back)
 * resolve to the first occurrence.
 */
export function zonedDateTimeToUtc(dateKey: string, minuteOfDay: number, tz: string): Date {
  const { year, month, day } = parseDateKey(dateKey);
  const dayCarry = Math.floor(minuteOfDay / MINUTES_PER_DAY);
  const minute = minuteOfDay - dayCarry * MINUTES_PER_DAY;
  const naiveUtc = Date.UTC(year, month - 1, day + dayCarry, Math.floor(minute / 60), minute % 60);

  const firstOffset = tzOffsetMinutes(new Date(naiveUtc), tz);
  const firstGuess = naiveUtc - firstOffset * MS_PER_MINUTE;
  const secondOffset = tzOffsetMinutes(new Date(firstGuess), tz);
  const secondGuess = naiveUtc - secondOffset * MS_PER_MINUTE;
  return new Date(secondGuess);
}

/** Add `n` calendar days to a dateKey (n may be negative). */
export function addDays(dateKey: string, n: number): string {
  const { year, month, day } = parseDateKey(dateKey);
  const d = new Date(Date.UTC(year, month - 1, day + n));
  return toDateKey(d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate());
}

/** Signed number of calendar days from `a` to `b` (b − a). */
export function daysBetween(a: string, b: string): number {
  const pa = parseDateKey(a);
  const pb = parseDateKey(b);
  const ua = Date.UTC(pa.year, pa.month - 1, pa.day);
  const ub = Date.UTC(pb.year, pb.month - 1, pb.day);
  return Math.round((ub - ua) / MS_PER_DAY);
}

/**
 * Format a minute-of-day as 24h "HH:MM". Values outside 0..1439 are wrapped,
 * so a past-midnight slot at 1500 renders as "01:00".
 */
export function formatMinuteOfDay(m: number): string {
  const wrapped = ((Math.round(m) % MINUTES_PER_DAY) + MINUTES_PER_DAY) % MINUTES_PER_DAY;
  return `${pad2(Math.floor(wrapped / 60))}:${pad2(wrapped % 60)}`;
}
