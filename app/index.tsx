import * as Haptics from 'expo-haptics';
import { Link, Redirect } from 'expo-router';
import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { BurningCigarette } from '../src/components/BurningCigarette';
import { GearIcon } from '../src/components/icons';
import { Body, Button, Chip, Dots, Eyebrow } from '../src/components/ui';
import { formatClock, formatCountdown, formatGap, formatLongDate, formatShortDate, formatWindowLeft } from '../src/domain/format';
import { quitDateKey, stageInfo } from '../src/domain/plan';
import { formatMinuteOfDay } from '../src/domain/time';
import type { Slot } from '../src/domain/types';
import { useNow } from '../src/hooks/useNow';
import { useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

export default function HomeScreen() {
  const settings = useAppStore((s) => s.settings);
  const plan = useAppStore((s) => s.plan);
  const todayLog = useAppStore((s) => s.todayLog);
  const subscription = useAppStore((s) => s.subscription);
  const markSlot = useAppStore((s) => s.markSlot);
  const now = useNow(1000);

  const slots = useMemo(() => todayLog?.slots ?? [], [todayLog]);
  const nowMs = now.getTime();

  // The "open" slot is the most recent one whose time has passed and that the
  // user hasn't answered yet. Earlier unanswered slots count as missed.
  const { openSlot, nextSlot, done } = useMemo(() => {
    let open: Slot | null = null;
    let next: Slot | null = null;
    for (const slot of slots) {
      const at = new Date(slot.scheduledAtIso).getTime();
      if (at <= nowMs) {
        if (slot.status === 'pending') open = slot;
      } else if (!next && slot.status === 'pending') {
        next = slot;
      }
    }
    return { openSlot: open, nextSlot: next, done: slots.map((s) => s.status === 'smoked') };
  }, [slots, nowMs]);

  if (subscription?.kind === 'expired') return <Redirect href="/paywall" />;
  if (!plan || !todayLog) return null;

  const stage = stageInfo(settings, plan, todayLog.dateKey);
  const smokedCount = slots.filter((s) => s.status === 'smoked').length;
  const quitKey = quitDateKey(settings, plan);

  const answer = (slot: Slot, status: 'smoked' | 'skipped') => {
    void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    void markSlot(todayLog.dateKey, slot.index, status);
  };

  // ---- Alert state: a slot is open right now -------------------------------
  if (openSlot) {
    // The window lasts one minimum gap; the cigarette burns down over it.
    const openedMs = new Date(openSlot.scheduledAtIso).getTime();
    const windowMs = settings.minGapMinutes * 60_000;
    const progress = (nowMs - openedMs) / windowMs;
    const closed = progress >= 1;
    return (
      <SafeAreaView style={[styles.screen, { backgroundColor: colors.teal }]}>
        <View style={styles.alertBody}>
          <Eyebrow light>Window open since {formatClock(openSlot.scheduledAtIso, settings.timeZone)}</Eyebrow>
          <Text style={styles.alertTitle}>{closed ? 'Window closed.' : 'You can smoke now.'}</Text>
          <Text style={styles.alertText}>
            {closed
              ? 'Still log it so your day stays honest. The next alert is on its way.'
              : `Or skip it. Skipped slots count toward your step-down. Either way, the next alert is at least ${formatGap(settings.minGapMinutes)} away.`}
          </Text>
        </View>
        <View style={{ alignItems: 'center', gap: 12 }}>
          <BurningCigarette progress={progress} />
          <Text style={[styles.caption, { color: '#FFFFFF' }]}>{formatWindowLeft(openedMs + windowMs - nowMs)}</Text>
        </View>
        <View style={{ alignItems: 'center', gap: 14, marginTop: 28 }}>
          <Dots total={slots.length} done={done} nextIndex={openSlot.index} onDark />
          <Text style={[styles.caption, { color: colors.tealLight }]}>
            Slot {openSlot.index + 1} of {slots.length} · Stage {stage.stageNumber}
          </Text>
        </View>
        <View style={{ flex: 1 }} />
        <View style={{ gap: 12 }}>
          <Button title="I smoked it" variant="onTeal" onPress={() => answer(openSlot, 'smoked')} />
          <Button title="Skip this one" variant="onTealOutline" onPress={() => answer(openSlot, 'skipped')} />
        </View>
      </SafeAreaView>
    );
  }

  // ---- Waiting / done states ----------------------------------------------
  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 28, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={{ gap: 6 }}>
            <Text style={styles.date}>{formatLongDate(todayLog.dateKey)}</Text>
            <Chip>
              {stage.isQuit
                ? 'Quit stage · 0 a day'
                : `Stage ${stage.stageNumber} · ${stage.perDay} a day · Day ${stage.dayInStage} of ${stage.daysPerStage}`}
            </Chip>
          </View>
          <Link href="/settings" asChild>
            <GearIcon.Button accessibilityLabel="Settings" />
          </Link>
        </View>

        <View style={styles.hero}>
          {stage.isQuit ? (
            <>
              <Eyebrow>Quit day</Eyebrow>
              <Text style={styles.heroTitle}>No alerts.</Text>
              <Text style={styles.heroSub}>You made it to zero. Keep the app for when a craving hits.</Text>
            </>
          ) : nextSlot ? (
            <>
              <Eyebrow>Next cigarette</Eyebrow>
              <Text style={styles.heroTime}>{formatClock(nextSlot.scheduledAtIso, settings.timeZone)}</Text>
              <Text style={styles.heroSub}>{formatCountdown(new Date(nextSlot.scheduledAtIso).getTime() - nowMs)}</Text>
            </>
          ) : (
            <>
              <Eyebrow>Done for today</Eyebrow>
              <Text style={styles.heroTitle}>That&apos;s all.</Text>
              <Text style={styles.heroSub}>Alerts start again tomorrow after {formatMinuteOfDay(settings.wakeMinutes)}.</Text>
            </>
          )}
        </View>

        {!stage.isQuit ? (
          <View style={{ alignItems: 'center', gap: 14 }}>
            <Dots total={slots.length} done={done} nextIndex={nextSlot?.index ?? null} />
            <Text style={styles.caption}>
              {smokedCount} of {slots.length} today
            </Text>
          </View>
        ) : null}

        {slots.length > 0 ? (
          <View style={styles.list}>
            {slots.map((slot, i) => {
              const passed = new Date(slot.scheduledAtIso).getTime() <= nowMs;
              const isNext = nextSlot?.index === slot.index;
              const label =
                slot.status === 'smoked' ? 'Smoked' : slot.status === 'skipped' ? 'Skipped' : passed ? 'Missed' : isNext ? 'Next' : 'Later';
              const tone =
                slot.status === 'smoked' ? colors.tealDeep : isNext ? colors.amberDeep : colors.muted;
              return (
                <View
                  key={slot.index}
                  style={[styles.listRow, i < slots.length - 1 && styles.listDivider, isNext && { backgroundColor: colors.amberSoft }]}
                >
                  <Text style={[styles.listTime, !passed && !isNext && { color: colors.muted }, isNext && { fontWeight: '800' }]}>
                    {formatClock(slot.scheduledAtIso, settings.timeZone)}
                  </Text>
                  <Text style={[styles.listStatus, { color: tone }]}>{label}</Text>
                </View>
              );
            })}
          </View>
        ) : null}

        <Body style={{ fontSize: 14, textAlign: 'center' }}>
          Alerts arrive at random times, at least {formatGap(settings.minGapMinutes)} apart.
        </Body>

        <Link href="/plan" asChild>
          <Button title={`See my plan · quit on ${formatShortDate(quitKey)}`} variant="secondary" />
        </Link>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  header: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' },
  date: { fontSize: 15, fontWeight: '600', color: colors.muted },
  hero: { alignItems: 'center', gap: 8, paddingTop: 36, paddingBottom: 20 },
  heroTime: { fontSize: 88, lineHeight: 92, fontWeight: '800', letterSpacing: -3, color: colors.ink, fontVariant: ['tabular-nums'] },
  heroTitle: { fontSize: 44, lineHeight: 48, fontWeight: '800', letterSpacing: -1, color: colors.ink },
  heroSub: { fontSize: 18, fontWeight: '600', color: colors.muted, textAlign: 'center' },
  caption: { fontSize: 15, fontWeight: '600', color: colors.muted },
  list: { borderWidth: 1, borderColor: colors.line, borderRadius: 20, overflow: 'hidden' },
  listRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 14, paddingHorizontal: 16 },
  listDivider: { borderBottomWidth: 1, borderBottomColor: colors.line },
  listTime: { fontSize: 16, fontWeight: '700', color: colors.ink, fontVariant: ['tabular-nums'] },
  listStatus: { fontSize: 14, fontWeight: '700' },
  alertBody: { alignItems: 'center', gap: 10, paddingTop: 56, paddingBottom: 28, paddingHorizontal: 8 },
  alertTitle: { fontSize: 44, lineHeight: 48, fontWeight: '800', letterSpacing: -1, color: '#FFFFFF', textAlign: 'center' },
  alertText: { fontSize: 17, lineHeight: 25, color: colors.tealLight, textAlign: 'center', maxWidth: 300 },
});
