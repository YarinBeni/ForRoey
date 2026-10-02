import { useRouter } from 'expo-router';
import React from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CloseIcon } from '../src/components/icons';
import { Title } from '../src/components/ui';
import { applyLanguage, LANGUAGE_NAMES, LANGUAGES, t, type LanguageSetting } from '../src/i18n';
import { useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

export default function LanguageScreen() {
  const router = useRouter();
  const current = useAppStore((s) => s.settings.language);
  const onboarded = useAppStore((s) => s.onboarded);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const setDraft = useAppStore((s) => s.setDraftSettings);

  const options: { value: LanguageSetting; label: string }[] = [
    { value: 'system', label: t('settings.languageSystem') },
    ...LANGUAGES.map((code) => ({ value: code, label: LANGUAGE_NAMES[code] })),
  ];

  const pick = async (language: LanguageSetting) => {
    // Apply first so rescheduled notifications already use the new language.
    const needsRestart = applyLanguage(language);
    if (onboarded) await updateSettings({ language });
    else setDraft({ language });
    router.back();
    if (needsRestart) Alert.alert(t('settings.restartAppTitle'), t('settings.restartAppBody'));
  };

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <View style={styles.header}>
        <Title size={28}>{t('settings.language')}</Title>
        <CloseIcon.Button accessibilityLabel={t('common.close')} onPress={() => router.back()} />
      </View>
      <View style={styles.list}>
        {options.map((option, i) => {
          const selected = option.value === current;
          return (
            <Pressable
              key={option.value}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              onPress={() => void pick(option.value)}
              style={({ pressed }) => [styles.row, i < options.length - 1 && styles.divider, pressed && { opacity: 0.7 }]}
            >
              <Text style={[styles.label, selected && { color: colors.tealDeep, fontWeight: '800' }]}>{option.label}</Text>
              {selected ? <Text style={styles.check}>✓</Text> : null}
            </Pressable>
          );
        })}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 },
  list: { borderWidth: 1, borderColor: colors.line, borderRadius: 16, overflow: 'hidden' },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', minHeight: 56, paddingHorizontal: 16 },
  divider: { borderBottomWidth: 1, borderBottomColor: colors.line },
  label: { fontSize: 17, fontWeight: '600', color: colors.ink },
  check: { fontSize: 18, fontWeight: '800', color: colors.tealDeep },
});
