import { Link, useRouter } from 'expo-router';
import React from 'react';
import { Alert, Platform, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { previewStatusIcon, statusIconSupported } from '../modules/cigarette-status';
import { BackIcon } from '../src/components/icons';
import { TimeField } from '../src/components/TimeField';
import { Button, Card, Row, SectionLabel, Stepper, Title } from '../src/components/ui';
import { formatGap } from '../src/domain/format';
import { LANGUAGE_NAMES, t } from '../src/i18n';
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

  const store = Platform.OS === 'ios' ? t('settings.appStore') : t('settings.playStore');
  const price = t('paywall.perMonth', { price: subscriptionService.priceLabel });

  const toggleNotifications = async (on: boolean) => {
    if (on) {
      const granted = await requestNotificationPermission();
      if (!granted) {
        Alert.alert(t('settings.allowTitle'), t('settings.allowBody'));
        return;
      }
    }
    await updateSettings({ notificationsEnabled: on });
  };

  const restore = async () => {
    try {
      await subscriptionService.restore();
      await refreshSubscription();
      Alert.alert(t('settings.restoredTitle'), t('settings.restoredBody'));
    } catch (e) {
      Alert.alert(t('settings.nothingToRestore'), e instanceof Error ? e.message : t('settings.tryLater'));
    }
  };

  const confirmRestart = () => {
    Alert.alert(t('settings.restartTitle'), t('settings.restartBody'), [
      { text: t('common.cancel'), style: 'cancel' },
      { text: t('settings.restart'), style: 'destructive', onPress: () => void restartPlan() },
    ]);
  };

  const subTitle =
    subscription?.kind === 'active' ? t('settings.subscribed') : subscription?.kind === 'trial' ? t('settings.freeTrial') : t('settings.trialEnded');
  const subHint =
    subscription?.kind === 'active'
      ? t('settings.manage', { price, store })
      : subscription?.kind === 'trial'
        ? t('settings.daysLeft', { count: subscription.daysLeft, price })
        : t('settings.subscribeToKeep');

  const languageLabel = settings.language === 'system' ? t('settings.languageSystem') : LANGUAGE_NAMES[settings.language];

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 24, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <BackIcon.Button accessibilityLabel={t('common.back')} onPress={() => router.back()} />
          <Title size={28}>{t('settings.title')}</Title>
        </View>

        <View style={{ gap: 10 }}>
          <SectionLabel>{t('settings.awakeHours')}</SectionLabel>
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <TimeField label={t('onboarding.wake')} value={settings.wakeMinutes} onChange={(v) => void updateSettings({ wakeMinutes: v })} />
            <TimeField label={t('onboarding.bed')} value={settings.sleepMinutes} onChange={(v) => void updateSettings({ sleepMinutes: v })} />
          </View>
          <Card>
            <Link href="/timezone" asChild>
              <Row label={t('onboarding.timeZone')} right={<Text style={styles.value}>{settings.timeZone}</Text>} onPress={() => {}} />
            </Link>
            <Link href="/language" asChild>
              <Row label={t('settings.language')} right={<Text style={styles.value}>{languageLabel}</Text>} onPress={() => {}} last />
            </Link>
          </Card>
        </View>

        <View style={{ gap: 10 }}>
          <SectionLabel>{t('settings.alerts')}</SectionLabel>
          <Card>
            <Row
              label={t('settings.minGap')}
              right={
                <Stepper
                  value={settings.minGapMinutes}
                  min={10}
                  max={240}
                  step={10}
                  format={formatGap}
                  onChange={(v) => void updateSettings({ minGapMinutes: v })}
                  labelLess={t('settings.minGap')}
                  labelMore={t('settings.minGap')}
                />
              }
            />
            <Row
              label={t('settings.notifications')}
              right={
                <Switch
                  value={settings.notificationsEnabled}
                  onValueChange={(v) => void toggleNotifications(v)}
                  trackColor={{ true: colors.teal, false: colors.line }}
                  accessibilityLabel={t('settings.notifications')}
                />
              }
              last={!statusIconSupported}
            />
            {statusIconSupported ? (
              <Row
                label={t('settings.statusBar')}
                hint={t('settings.statusBarHint')}
                right={<Text style={styles.link}>{t('settings.preview')}</Text>}
                onPress={() => previewStatusIcon(2 * 60_000)}
                last
              />
            ) : null}
          </Card>
        </View>

        <View style={{ gap: 10 }}>
          <SectionLabel>{t('settings.subscription')}</SectionLabel>
          <View style={styles.subCard}>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <View style={{ flex: 1, gap: 2 }}>
                <Text style={styles.subTitle}>{subTitle}</Text>
                <Text style={styles.subHint}>{subHint}</Text>
              </View>
              {subscription?.kind !== 'active' ? (
                <Link href="/paywall" asChild>
                  <Button title={t('settings.subscribe')} style={{ height: 44, borderRadius: 12 }} />
                </Link>
              ) : null}
            </View>
            <Button title={t('settings.restore')} variant="ghost" style={{ alignSelf: 'flex-start', paddingHorizontal: 0 }} onPress={restore} />
          </View>
        </View>

        <Button title={t('settings.restartPlan')} variant="danger" onPress={confirmRestart} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  value: { fontSize: 15, fontWeight: '700', color: colors.muted },
  link: { fontSize: 15, fontWeight: '700', color: colors.teal },
  subCard: { gap: 8, padding: 16, borderRadius: 16, backgroundColor: colors.surface },
  subTitle: { fontSize: 16, fontWeight: '800', color: colors.ink },
  subHint: { fontSize: 14, color: colors.muted },
});
