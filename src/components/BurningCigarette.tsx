import React from 'react';
import { StyleSheet, View } from 'react-native';

/**
 * A cigarette that burns down as the smoking window passes.
 *
 * `progress` runs from 0 (window just opened, whole cigarette) to 1 (window
 * closed, burnt down to the filter). Drawn with plain Views so it needs no
 * SVG or image assets.
 */
export function BurningCigarette({ progress, width = 280 }: { progress: number; width?: number }) {
  const p = Math.min(1, Math.max(0, progress));
  const filterWidth = 64;
  const emberWidth = 12;
  const maxPaper = width - filterWidth - emberWidth;
  const paperWidth = Math.round(maxPaper * (1 - p));
  // Ash grows a little as the paper burns, then falls off.
  const ashWidth = p > 0 && p < 1 ? Math.min(16, Math.round(maxPaper * p * 0.25)) : 0;
  const out = p >= 1;

  return (
    <View
      accessibilityRole="progressbar"
      accessibilityLabel="Smoking window"
      accessibilityValue={{ min: 0, max: 100, now: Math.round(p * 100) }}
      style={[styles.row, { width }]}
    >
      <View style={[styles.filter, { width: filterWidth }]}>
        <View style={styles.band} />
      </View>
      <View style={[styles.paper, { width: paperWidth }]} />
      {ashWidth > 0 ? <View style={[styles.ash, { width: ashWidth }]} /> : null}
      <View style={[styles.emberHalo, out && styles.emberOut]}>
        <View style={[styles.ember, out && styles.emberOutCore]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', height: 32 },
  filter: {
    height: 26,
    backgroundColor: '#D9A86C',
    borderTopLeftRadius: 13,
    borderBottomLeftRadius: 13,
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  band: { width: 3, height: 26, backgroundColor: '#B8863F' },
  paper: { height: 26, backgroundColor: '#FFFFFF' },
  ash: { height: 22, backgroundColor: '#B9BDBB', borderRadius: 3 },
  emberHalo: {
    width: 20,
    height: 20,
    marginLeft: -4,
    borderRadius: 10,
    backgroundColor: 'rgba(255, 177, 92, 0.55)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  ember: { width: 12, height: 12, borderRadius: 6, backgroundColor: '#FF7A1A' },
  emberOut: { backgroundColor: 'rgba(0, 0, 0, 0.12)' },
  emberOutCore: { backgroundColor: '#7B8582' },
});
