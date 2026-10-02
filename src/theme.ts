/**
 * Design tokens for Pacer.
 *
 * Kept as plain objects (no React Native imports) so they can be used from
 * anywhere, including tests and non-UI modules.
 */

export const colors = {
  /** Page background. */
  ground: '#FFFFFF',
  /** Cards, inputs and other raised surfaces. */
  surface: '#F2F4F3',
  /** Primary text. */
  ink: '#15201D',
  /** Secondary text. */
  muted: '#5B6B67',
  /** Hairlines and dividers. */
  line: '#DDE3E1',
  /** Brand / primary action colour. */
  teal: '#157A6E',
  /** Tinted background for teal content. */
  tealSoft: '#E3F1EE',
  /** Darker teal for text on light tints and the white-on-teal button label. */
  tealDeep: '#0F5F55',
  /** Light teal for secondary text on a teal background. */
  tealLight: '#CFE6E1',
  /** Warm accent used for "it's time" states. */
  amber: '#C96F1C',
  /** Tinted background for amber content. */
  amberSoft: '#FBEEDD',
  /** Darker amber for small text (meets 4.5:1 on white). */
  amberDeep: '#9A5410',
  /** Destructive actions and errors. */
  danger: '#B3362B',
} as const;

export type ColorToken = keyof typeof colors;

export const radius = {
  sm: 10,
  md: 16,
  lg: 24,
} as const;

/** 4-point spacing scale. */
export const spacing = {
  xxs: 4,
  xs: 8,
  sm: 12,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

/** Font sizes (points). Line heights are roughly 1.3x. */
export const font = {
  caption: 12,
  small: 14,
  body: 16,
  title: 20,
  heading: 28,
  display: 40,
} as const;

export const theme = { colors, radius, spacing, font } as const;
export type Theme = typeof theme;
