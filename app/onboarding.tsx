import { Link } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { TimeField } from '../src/components/TimeField';
import { Body, Button, Chip, Eyebrow, SectionLabel, Stepper, Title } from '../src/components/ui';
import { formatGap, tzOffsetLabel } from '../src/domain/format';
import { awakeWindowMinutes } from '../src/domain/schedule';
import { DEFAULT_SETTINGS } from '../src/domain/types';
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
      if (!granted) {
        Alert.alert(
          'Notifications are off',
          'Pacer works by sending you alerts. You can turn them on later in Settings, and the plan will still track your day.',
        );
      }
      await completeOnboarding({ ...draft, notificationsEnabled: granted });
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 24, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={{ gap: 10 }}>
          <Eyebrow>Set up</Eyebrow>
          <Title>When are you awake?</Title>
          <Body>Pacer only sends cigarette alerts inside this window, in your time zone.</Body>
        </View>

        <View style={{ flexDirection: 'row', gap: 12 }}>
          <TimeField label="Wake up" value={draft.wakeMinutes} onChange={(v) => setDraft({ wakeMinutes: v })} />
          <TimeField label="Bed" value={draft.sleepMinutes} onChange={(v) => setDraft({ sleepMinutes: v })} />
        </View>

        <Link href="/timezone" asChild>
          <Pressable accessibilityRole="button" style={({ pressed }) => [styles.tzRow, pressed && { opacity: 0.7 }]}>
            <View style={{ gap: 2 }}>
              <Text style={styles.tzLabel}>Time zone</Text>
              <Text style={styles.tzValue}>
                {draft.timeZone} · {tzOffsetLabel(draft.timeZone)}
              </Text>
            </View>
            <Chip>{draft.timeZone === DEFAULT_SETTINGS.timeZone ? 'Detected' : 'Custom'}</Chip>
          </Pressable>
        </Link>

        <View style={{ gap: 12 }}>
          <SectionLabel>Starting point</SectionLabel>
          <View style={styles.settingRow}>
            <View style={{ flex: 1, gap: 2 }}>
              <Text style={styles.settingLabel}>Cigarettes per day</Text>
              <Text style={styles.settingHint}>Drops by one every {draft.daysPerStage} days</Text>
            </View>
            <Stepper
              value={draft.startPerDay}
              min={1}
              max={Math.min(20, maxFeasible)}
              onChange={(v) => setDraft({ startPerDay: v })}
              labelLess="Fewer per day"
              labelMore="More per day"
            />
          </View>
          <View style={styles.settingRow}>
            <View style={{ flex: 1, gap: 2 }}>
              <Text style={styles.settingLabel}>Minimum gap</Text>
              <Text style={styles.settingHint}>Between two alerts</Text>
            </View>
            <Stepper
              value={draft.minGapMinutes}
              min={10}
              max={240}
              step={10}
              format={formatGap}
              onChange={(v) => setDraft({ minGapMinutes: v })}
              labelLess="Shorter gap"
              labelMore="Longer gap"
            />
          </View>
        </View>

        <Body style={{ fontSize: 14 }}>
          {awakeHours} awake hours, {draft.startPerDay} alerts. Times are random every day and always at least{' '}
          {formatGap(draft.minGapMinutes)} apart.
        </Body>
      </ScrollView>
      <Button title={busy ? 'Building…' : 'Build my plan'} disabled={busy} onPress={finish} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.md },
  tzRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
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
    paddingLeft: 16,
    paddingRight: 12,
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: 16,
  },
  settingLabel: { fontSize: 16, fontWeight: '700', color: colors.ink },
  settingHint: { fontSize: 13, color: colors.muted },
});
