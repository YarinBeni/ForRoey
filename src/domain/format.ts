import { localeTag, t } from '../i18n';

/** Human-friendly, translated formatting helpers used by the screens. */

export function formatCountdown(ms: number): string {
  if (ms <= 0) return t('time.now');
  const totalMin = Math.round(ms / 60_000);
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  if (h === 0) return t('time.inMin', { m: Math.max(1, m) });
  if (m === 0) return t('time.inH', { h });
  return t('time.inHM', { h, m });
}

/** "Window closes in 22 min" / "Window closed". */
export function formatWindowLeft(ms: number): string {
  if (ms <= 0) return t('time.windowClosed');
  const totalSec = Math.ceil(ms / 1000);
  if (totalSec < 60) return t('time.windowClosesSec', { s: totalSec });
  return t('time.windowClosesMin', { m: Math.ceil(totalSec / 60) });
}

export function formatGap(minutes: number): string {
  if (minutes < 60) return t('time.gapMin', { m: minutes });
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m === 0 ? t('time.gapH', { h }) : t('time.gapHM', { h, m });
}

function dateFromKey(dateKey: string): Date {
  const [y, m, d] = dateKey.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d, 12));
}

function intlDate(options: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  try {
    return new Intl.DateTimeFormat(localeTag(), options);
  } catch {
    return new Intl.DateTimeFormat('en-GB', options);
  }
}

export function formatLongDate(dateKey: string): string {
  return intlDate({ weekday: 'long', day: 'numeric', month: 'long', timeZone: 'UTC' }).format(dateFromKey(dateKey));
}

export function formatShortDate(dateKey: string): string {
  return intlDate({ day: 'numeric', month: 'short', timeZone: 'UTC' }).format(dateFromKey(dateKey));
}

export function formatClock(iso: string, timeZone: string): string {
  return intlDate({ hour: '2-digit', minute: '2-digit', hour12: false, timeZone }).format(new Date(iso));
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
