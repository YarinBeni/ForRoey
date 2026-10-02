/** Human-friendly formatting helpers used by the screens. */

export function formatCountdown(ms: number): string {
  if (ms <= 0) return 'now';
  const totalMin = Math.round(ms / 60_000);
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  if (h === 0) return `in ${Math.max(1, m)} min`;
  if (m === 0) return `in ${h} h`;
  return `in ${h} h ${m} min`;
}

export function formatLongDate(dateKey: string, locale = 'en-GB'): string {
  const [y, m, d] = dateKey.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d, 12));
  return new Intl.DateTimeFormat(locale, { weekday: 'long', day: 'numeric', month: 'long', timeZone: 'UTC' }).format(date);
}

export function formatShortDate(dateKey: string, locale = 'en-GB'): string {
  const [y, m, d] = dateKey.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d, 12));
  return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short', timeZone: 'UTC' }).format(date);
}

export function formatClock(iso: string, timeZone: string, locale = 'en-GB'): string {
  return new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit', hour12: false, timeZone }).format(new Date(iso));
}

export function formatGap(minutes: number): string {
  if (minutes < 60) return `${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

/** "GMT+3" style offset label for a time zone right now. */
export function tzOffsetLabel(timeZone: string, at = new Date()): string {
  try {
    const parts = new Intl.DateTimeFormat('en-US', { timeZone, timeZoneName: 'shortOffset' }).formatToParts(at);
    return parts.find((p) => p.type === 'timeZoneName')?.value ?? '';
  } catch {
    return '';
  }
}

/** "Window closes in 22 min" / "Window closed". */
export function formatWindowLeft(ms: number): string {
  if (ms <= 0) return 'Window closed';
  const totalSec = Math.ceil(ms / 1000);
  if (totalSec < 60) return `Window closes in ${totalSec} s`;
  const min = Math.ceil(totalSec / 60);
  return `Window closes in ${min} min`;
}
