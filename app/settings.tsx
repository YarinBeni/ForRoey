import { Link, useRouter } from 'expo-router';
import React from 'react';
import { Alert, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { BackIcon } from '../src/components/icons';
import { TimeField } from '../src/components/TimeField';
import { Button, Card, Row, SectionLabel, Stepper, Title } from '../src/components/ui';
import { formatGap } from '../src/domain/format';
import { requestNotificationPermission } from '../src/services/notifications';
import { getSubscriptionService, useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

const subscriptionService = getSubscriptionService();

export default function SettingsScreen() {
  const router = useRouter();
  const settings = useAppStore((s) => s.settings);
  const subscription = useAppStore((s) => s.subscription);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const restartPlan = useAppStore((s) => s.restartPlan);
  const refreshSubscription = useAppStore((s) => s.refreshSubscription);
  if (!settings) return null;

  const toggleNotifications = async (on: boolean) => {
    if (on) {
      const granted = await requestNotificationPermission();
      if (!granted) {
        Alert.alert('Allow notifications', 'Turn on notifications for Pacer in the iOS Settings app, then come back here.');
        return;
      }
    }
    await updateSettings({ notificationsEnabled: on });
  };

  const restore = async () => {
    try {
      await subscriptionService.restore();
      await refreshSubscription();
      Alert.alert('Restored', 'Your purchases are up to date.');
    } catch (e) {
      Alert.alert('Nothing to restore', e instanceof Error ? e.message : 'Try again later.');
    }
  };

  const confirmRestart = () => {
    Alert.alert('Restart your plan?', 'Today becomes day 1 of stage 1 again. Your settings stay as they are.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Restart', style: 'destructive', onPress: () => void restartPlan() },
    ]);
  };

  const subTitle =
    subscription?.kind === 'active' ? 'Subscribed' : subscription?.kind === 'trial' ? 'Free trial' : 'Trial ended';
  const subHint =
    subscription?.kind === 'active'
      ? `${subscriptionService.priceLabel} · manage in the App Store`
      : subscription?.kind === 'trial'
        ? `${subscription.daysLeft} days left, then ${subscriptionService.priceLabel}`
        : `Subscribe to keep your alerts going`;

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 24, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <BackIcon.Button accessibilityLabel="Back to home" onPress={() => router.back()} />
          <Title size={28}>Settings</Title>
        </View>

        <View style={{ gap: 10 }}>
          <SectionLabel>Awake hours</SectionLabel>
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <TimeField label="Wake up" value={settings.wakeMinutes} onChange={(v) => void updateSettings({ wakeMinutes: v })} />
            <TimeField label="Bed" value={settings.sleepMinutes} onChange={(v) => void updateSettings({ sleepMinutes: v })} />
          </View>
          <Card>
            <Link href="/timezone" asChild>
              <Row label="Time zone" right={<Text style={styles.value}>{settings.timeZone}</Text>} onPress={() => {}} last />
            </Link>
          </Card>
        </View>

        <View style={{ gap: 10 }}>
          <SectionLabel>Alerts</SectionLabel>
          <Card>
            <Row
              label="Minimum gap"
              right={
                <Stepper
                  value={settings.minGapMinutes}
                  min={10}
                  max={240}
                  step={10}
                  format={formatGap}
                  onChange={(v) => void updateSettings({ minGapMinutes: v })}
                  labelLess="Shorter gap"
                  labelMore="Longer gap"
                />
              }
            />
            <Row
              label="Notifications"
              right={
                <Switch
                  value={settings.notificationsEnabled}
                  onValueChange={(v) => void toggleNotifications(v)}
                  trackColor={{ true: colors.teal, false: colors.line }}
                  accessibilityLabel="Notifications"
                />
              }
              last
            />
          </Card>
        </View>

        <View style={{ gap: 10 }}>
          <SectionLabel>Subscription</SectionLabel>
          <View style={styles.subCard}>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <View style={{ flex: 1, gap: 2 }}>
                <Text style={styles.subTitle}>{subTitle}</Text>
                <Text style={styles.subHint}>{subHint}</Text>
              </View>
              {subscription?.kind !== 'active' ? (
                <Link href="/paywall" asChild>
                  <Button title="Subscribe" style={{ height: 44, borderRadius: 12 }} />
                </Link>
              ) : null}
            </View>
            <Button title="Restore purchases" variant="ghost" style={{ alignSelf: 'flex-start', paddingHorizontal: 0 }} onPress={restore} />
          </View>
        </View>

        <Button title="Restart my plan from today" variant="danger" onPress={confirmRestart} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  value: { fontSize: 15, fontWeight: '700', color: colors.muted },
  subCard: { gap: 8, padding: 16, borderRadius: 16, backgroundColor: colors.surface },
  subTitle: { fontSize: 16, fontWeight: '800', color: colors.ink },
  subHint: { fontSize: 14, color: colors.muted },
});
