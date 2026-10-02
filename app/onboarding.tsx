import { Link } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { TimeField } from '../src/components/TimeField';
import { Body, Button, Chip, Eyebrow, SectionLabel, Stepper, Title } from '../src/components/ui';
import { formatGap, tzOffsetLabel } from '../src/domain/format';
import { awakeWindowMinutes } from '../src/domain/schedule';
import { DEFAULT_SETTINGS } from '../src/domain/types';
import { LANGUAGE_NAMES, resolveLanguage, t } from '../src/i18n';
import { requestNotificationPermission } from '../src/services/notifications';
import { useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

export default function OnboardingScreen() {
  const completeOnboarding = useAppStore((s) => s.completeOnboarding);
  const draft = useAppStore((s) => s.settings) ?? DEFAULT_SETTINGS;
  const setDraft = useAppStore((s) => s.setDraftSettings);
  const [busy, setBusy] = useState(false);

  const awakeHours = Math.round(awakeWindowMinutes(draft.wakeMinutes, draft.sleepMinutes) / 60);
  const maxFeasible = Math.max(1, Math.floor(awakeWindowMinutes(draft.wakeMinutes, draft.sleepMinutes) / draft.minGapMinutes));

  const finish = async () => {
    setBusy(true);
    try {
      const granted = await requestNotificationPermission();
      if (!granted) Alert.alert(t('onboarding.notifOffTitle'), t('onboarding.notifOffBody'));
      await completeOnboarding({ ...draft, notificationsEnabled: granted });
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 24, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={styles.topRow}>
          <Eyebrow>{t('onboarding.eyebrow')}</Eyebrow>
          <Link href="/language" asChild>
            <Pressable accessibilityRole="button" accessibilityLabel={t('settings.language')} hitSlop={8}>
              <Text style={styles.langLink}>{LANGUAGE_NAMES[resolveLanguage(draft.language)]}</Text>
            </Pressable>
          </Link>
        </View>
        <View style={{ gap: 10 }}>
          <Title>{t('onboarding.title')}</Title>
          <Body>{t('onboarding.subtitle')}</Body>
        </View>

        <View style={{ flexDirection: 'row', gap: 12 }}>
          <TimeField label={t('onboarding.wake')} value={draft.wakeMinutes} onChange={(v) => setDraft({ wakeMinutes: v })} />
          <TimeField label={t('onboarding.bed')} value={draft.sleepMinutes} onChange={(v) => setDraft({ sleepMinutes: v })} />
        </View>

        <Link href="/timezone" asChild>
          <Pressable accessibilityRole="button" style={({ pressed }) => [styles.tzRow, pressed && { opacity: 0.7 }]}>
            <View style={{ gap: 2, flex: 1 }}>
              <Text style={styles.tzLabel}>{t('onboarding.timeZone')}</Text>
              <Text style={styles.tzValue}>
                {draft.timeZone} · {tzOffsetLabel(draft.timeZone)}
              </Text>
            </View>
            <Chip>{draft.timeZone === DEFAULT_SETTINGS.timeZone ? t('onboarding.detected') : t('onboarding.custom')}</Chip>
          </Pressable>
        </Link>

        <View style={{ gap: 12 }}>
          <SectionLabel>{t('onboarding.startingPoint')}</SectionLabel>
          <View style={styles.settingRow}>
            <View style={{ flex: 1, gap: 2 }}>
              <Text style={styles.settingLabel}>{t('onboarding.perDay')}</Text>
              <Text style={styles.settingHint}>{t('onboarding.perDayHint', { count: draft.daysPerStage })}</Text>
            </View>
            <Stepper
              value={draft.startPerDay}
              min={1}
              max={Math.min(20, maxFeasible)}
              onChange={(v) => setDraft({ startPerDay: v })}
              labelLess={t('onboarding.perDay')}
              labelMore={t('onboarding.perDay')}
            />
          </View>
          <View style={styles.settingRow}>
            <View style={{ flex: 1, gap: 2 }}>
              <Text style={styles.settingLabel}>{t('onboarding.minGap')}</Text>
              <Text style={styles.settingHint}>{t('onboarding.minGapHint')}</Text>
            </View>
            <Stepper
              value={draft.minGapMinutes}
              min={10}
              max={240}
              step={10}
              format={formatGap}
              onChange={(v) => setDraft({ minGapMinutes: v })}
              labelLess={t('onboarding.minGap')}
              labelMore={t('onboarding.minGap')}
            />
          </View>
        </View>

        <Body style={{ fontSize: 14 }}>
          {t('onboarding.summary', { hours: awakeHours, count: draft.startPerDay, gap: formatGap(draft.minGapMinutes) })}
        </Body>
      </ScrollView>
      <Button title={busy ? t('onboarding.building') : t('onboarding.build')} disabled={busy} onPress={finish} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.md },
  topRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  langLink: { fontSize: 14, fontWeight: '700', color: colors.teal, paddingVertical: 8 },
  tzRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 16,
  },
  tzLabel: { fontSize: 13, fontWeight: '600', color: colors.muted },
  tzValue: { fontSize: 16, fontWeight: '700', color: colors.ink },
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
