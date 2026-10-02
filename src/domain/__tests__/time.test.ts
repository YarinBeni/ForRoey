import {
  addDays,
  dateKeyInTz,
  daysBetween,
  formatMinuteOfDay,
  minuteOfDayInTz,
  tzOffsetMinutes,
  zonedDateTimeToUtc,
} from '../time';

const TZ = 'Asia/Jerusalem';

describe('zonedDateTimeToUtc', () => {
  it('converts a summer (IDT, +3) wall-clock time to UTC', () => {
    // 2026-10-02 08:00 Jerusalem = 05:00Z (DST ends late October).
    expect(zonedDateTimeToUtc('2026-10-02', 8 * 60, TZ).toISOString()).toBe('2026-10-02T05:00:00.000Z');
  });

  it('converts a winter (IST, +2) wall-clock time to UTC', () => {
    expect(zonedDateTimeToUtc('2026-01-15', 8 * 60, TZ).toISOString()).toBe('2026-01-15T06:00:00.000Z');
    expect(tzOffsetMinutes(new Date('2026-01-15T12:00:00Z'), TZ)).toBe(120);
    expect(tzOffsetMinutes(new Date('2026-07-15T12:00:00Z'), TZ)).toBe(180);
  });

  it('carries minutes >= 1440 into the next day', () => {
    // 25:30 on Oct 2 == 01:30 on Oct 3 Jerusalem == 22:30Z on Oct 2.
    expect(zonedDateTimeToUtc('2026-10-02', 1440 + 90, TZ).toISOString()).toBe('2026-10-02T22:30:00.000Z');
    expect(zonedDateTimeToUtc('2026-10-03', 90, TZ).getTime()).toBe(
      zonedDateTimeToUtc('2026-10-02', 1440 + 90, TZ).getTime(),
    );
  });

  it('round-trips through dateKeyInTz / minuteOfDayInTz', () => {
    const instant = zonedDateTimeToUtc('2026-03-27', 10 * 60 + 15, TZ);
    expect(dateKeyInTz(instant, TZ)).toBe('2026-03-27');
    expect(minuteOfDayInTz(instant, TZ)).toBe(10 * 60 + 15);
  });

  it('handles midnight and UTC', () => {
    expect(zonedDateTimeToUtc('2026-10-02', 0, 'UTC').toISOString()).toBe('2026-10-02T00:00:00.000Z');
    expect(dateKeyInTz(new Date('2026-10-02T23:30:00Z'), TZ)).toBe('2026-10-03');
    expect(minuteOfDayInTz(new Date('2026-10-02T21:00:00Z'), TZ)).toBe(0);
  });
});

describe('dateKey arithmetic', () => {
  it('adds days across month and year boundaries', () => {
    expect(addDays('2026-01-30', 2)).toBe('2026-02-01');
    expect(addDays('2026-12-31', 1)).toBe('2027-01-01');
    expect(addDays('2026-03-01', -1)).toBe('2026-02-28');
  });

  it('computes signed day differences', () => {
    expect(daysBetween('2026-10-02', '2026-10-09')).toBe(7);
    expect(daysBetween('2026-10-09', '2026-10-02')).toBe(-7);
    expect(daysBetween('2026-10-02', '2026-10-02')).toBe(0);
  });

  it('rejects malformed keys', () => {
    expect(() => addDays('2026/10/02', 1)).toThrow();
  });
});

describe('formatMinuteOfDay', () => {
  it('formats as 24h HH:MM and wraps past-midnight values', () => {
    expect(formatMinuteOfDay(8 * 60)).toBe('08:00');
    expect(formatMinuteOfDay(23 * 60 + 5)).toBe('23:05');
    expect(formatMinuteOfDay(0)).toBe('00:00');
    expect(formatMinuteOfDay(1440 + 60)).toBe('01:00');
  });
});
