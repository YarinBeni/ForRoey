import { useRouter } from 'expo-router';
import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { BackIcon } from '../src/components/icons';
import { Stepper, Title } from '../src/components/ui';
import { formatLongDate, formatShortDate } from '../src/domain/format';
import { quitDateKey, stageCounts, stageInfo } from '../src/domain/plan';
import { addDays, dateKeyInTz } from '../src/domain/time';
import { t } from '../src/i18n';
import { useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

export default function PlanScreen() {
  const router = useRouter();
  const settings = useAppStore((s) => s.settings);
  const plan = useAppStore((s) => s.plan);
  const updateSettings = useAppStore((s) => s.updateSettings);
  if (!plan) return null;

  const todayKey = dateKeyInTz(new Date(), settings.timeZone);
  const today = stageInfo(settings, plan, todayKey);
  const counts = stageCounts(settings.startPerDay);
  const quitKey = quitDateKey(settings, plan);

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 24, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <BackIcon.Button accessibilityLabel={t('common.back')} onPress={() => router.back()} />
          <Title size={28}>{t('plan.title')}</Title>
        </View>

        <View style={styles.quitCard}>
          <Text style={styles.quitEyebrow}>{t('plan.quitDate')}</Text>
          <Text style={styles.quitDate}>{formatLongDate(quitKey)}</Text>
          <Text style={styles.quitSub}>{t('plan.quitSub', { count: settings.daysPerStage, date: formatShortDate(plan.startDateKey) })}</Text>
        </View>

        <View style={{ gap: 10 }}>
          {counts.map((perDay, i) => {
            const startKey = addDays(plan.startDateKey, i * settings.daysPerStage);
            const endKey = addDays(startKey, settings.daysPerStage - 1);
            const isCurrent = !today.isQuit && today.stageNumber === i + 1;
            const isPast = today.isQuit || today.stageNumber > i + 1;
            return (
              <View key={perDay} style={[styles.stage, isCurrent && styles.stageCurrent]}>
                <View style={[styles.stageBadge, (isCurrent || isPast) && { backgroundColor: colors.teal }]}>
                  <Text style={[styles.stageBadgeText, (isCurrent || isPast) && { color: '#FFFFFF' }]}>{perDay}</Text>
                </View>
                <View style={{ flex: 1, gap: 6 }}>
                  <View style={styles.stageLine}>
                    <Text style={[styles.stageName, isCurrent && { fontWeight: '800' }]}>{t('plan.perDay', { n: perDay })}</Text>
                    <Text style={styles.stageDates}>
                      {formatShortDate(startKey)} – {formatShortDate(endKey)}
                    </Text>
                  </View>
                  {isCurrent ? (
                    <>
                      <View style={styles.bar}>
                        <View style={[styles.barFill, { width: `${Math.round((today.dayInStage / settings.daysPerStage) * 100)}%` }]} />
                      </View>
                      <Text style={styles.stageDates}>{t('plan.dayOf', { day: today.dayInStage, days: settings.daysPerStage })}</Text>
                    </>
                  ) : null}
                </View>
              </View>
            );
          })}
        </View>

        <View style={styles.settingRow}>
          <View style={{ flex: 1, gap: 2 }}>
            <Text style={styles.settingLabel}>{t('plan.daysPerStage')}</Text>
            <Text style={styles.settingHint}>{t('plan.daysPerStageHint')}</Text>
          </View>
          <Stepper
            value={settings.daysPerStage}
            min={1}
            max={60}
            onChange={(v) => void updateSettings({ daysPerStage: v })}
            labelLess={t('plan.daysPerStage')}
            labelMore={t('plan.daysPerStage')}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  quitCard: { gap: 4, padding: 18, borderRadius: 20, backgroundColor: colors.tealSoft },
  quitEyebrow: { fontSize: 13, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase', color: colors.tealDeep },
  quitDate: { fontSize: 30, fontWeight: '800', letterSpacing: -0.5, color: colors.tealDeep },
  quitSub: { fontSize: 14, fontWeight: '600', color: colors.tealDeep },
  stage: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.line,
  },
  stageCurrent: { borderWidth: 2, borderColor: colors.teal },
  stageBadge: { width: 44, height: 44, borderRadius: 12, backgroundColor: colors.surface, alignItems: 'center', justifyContent: 'center' },
  stageBadgeText: { fontSize: 20, fontWeight: '800', color: colors.ink },
  stageLine: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8 },
  stageName: { fontSize: 15, fontWeight: '700', color: colors.ink },
  stageDates: { fontSize: 13, fontWeight: '600', color: colors.muted },
  bar: { height: 6, borderRadius: 999, backgroundColor: colors.line, overflow: 'hidden' },
  barFill: { height: 6, backgroundColor: colors.teal },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 12,
    paddingStart: 16,
    paddingEnd: 12,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 16,
  },
  settingLabel: { fontSize: 16, fontWeight: '700', color: colors.ink },
  settingHint: { fontSize: 13, color: colors.muted },
});
