import { getLocales } from 'expo-localization';
import { I18n, type TranslateOptions } from 'i18n-js';
import { I18nManager } from 'react-native';

import am from './am';
import ar from './ar';
import en from './en';
import he from './he';
import ru from './ru';

/** Languages the app ships with. */
export const LANGUAGES = ['en', 'he', 'ar', 'ru', 'am'] as const;
export type Language = (typeof LANGUAGES)[number];
/** What the user picks: a language, or follow the phone. */
export type LanguageSetting = Language | 'system';

/** Each language's name in itself, for the picker. */
export const LANGUAGE_NAMES: Record<Language, string> = {
  en: 'English',
  he: 'עברית',
  ar: 'العربية',
  ru: 'Русский',
  am: 'አማርኛ',
};

const RTL_LANGUAGES: readonly Language[] = ['he', 'ar'];

/** BCP 47 tags used for Intl date/number formatting. Arabic keeps Western digits. */
const LOCALE_TAGS: Record<Language, string> = {
  en: 'en-GB',
  he: 'he-IL',
  ar: 'ar-u-nu-latn',
  ru: 'ru-RU',
  am: 'am-ET',
};

const i18n = new I18n({ en, he, ar, ru, am });
i18n.defaultLocale = 'en';
i18n.enableFallback = true;

// Russian: 1 день / 2-4 дня / 5+ дней.
i18n.pluralization.register('ru', (_i18n, count) => {
  const n = Math.abs(Math.trunc(count));
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return ['one', 'other'];
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return ['few', 'other'];
  return ['many', 'other'];
});

// Arabic: zero / one / two / 3-10 few / 11-99 many / other.
i18n.pluralization.register('ar', (_i18n, count) => {
  const n = Math.abs(Math.trunc(count));
  if (n === 0) return ['zero', 'other'];
  if (n === 1) return ['one', 'other'];
  if (n === 2) return ['two', 'other'];
  const mod100 = n % 100;
  if (mod100 >= 3 && mod100 <= 10) return ['few', 'other'];
  if (mod100 >= 11 && mod100 <= 99) return ['many', 'other'];
  return ['other'];
});

function isLanguage(code: string | null | undefined): code is Language {
  return !!code && (LANGUAGES as readonly string[]).includes(code);
}

/** The phone's language, if we ship it; English otherwise. */
export function deviceLanguage(): Language {
  try {
    for (const locale of getLocales()) {
      const code = locale.languageCode?.toLowerCase();
      // Older Android reports Hebrew as "iw".
      const normalized = code === 'iw' ? 'he' : code;
      if (isLanguage(normalized)) return normalized;
    }
  } catch {
    // expo-localization unavailable (tests): fall through.
  }
  return 'en';
}

export function resolveLanguage(setting: LanguageSetting): Language {
  return setting === 'system' ? deviceLanguage() : setting;
}

export function currentLanguage(): Language {
  return isLanguage(i18n.locale) ? i18n.locale : 'en';
}

export function isRtl(language: Language = currentLanguage()): boolean {
  return RTL_LANGUAGES.includes(language);
}

/** Locale tag for `Intl` formatting in the current language. */
export function localeTag(): string {
  return LOCALE_TAGS[currentLanguage()];
}

/**
 * Switch the UI language. Returns true when the layout direction changed,
 * which React Native only applies after the app restarts.
 */
export function applyLanguage(setting: LanguageSetting): boolean {
  const language = resolveLanguage(setting);
  i18n.locale = language;
  const rtl = isRtl(language);
  I18nManager.allowRTL(true);
  if (I18nManager.isRTL !== rtl) {
    I18nManager.forceRTL(rtl);
    return true;
  }
  return false;
}

/** Translate a key such as "home.nextCigarette", with %{name} interpolation and `count` plurals. */
export function t(key: string, options?: TranslateOptions): string {
  return i18n.t(key, options);
}
