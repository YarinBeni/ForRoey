import am from '../am';
import ar from '../ar';
import en from '../en';
import he from '../he';
import ru from '../ru';

type Tree = { [key: string]: string | Tree };

function flatten(tree: Tree, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'string') out[path] = value;
    else Object.assign(out, flatten(value, path));
  }
  return out;
}

const PLURAL_FORMS = new Set(['zero', 'one', 'two', 'few', 'many', 'other']);

/** Keys with plural forms collapsed, so locales may add forms English lacks. */
function structuralKeys(tree: Tree): string[] {
  return Object.keys(flatten(tree))
    .map((k) => {
      const parts = k.split('.');
      return PLURAL_FORMS.has(parts[parts.length - 1]) ? parts.slice(0, -1).join('.') + '.*' : k;
    })
    .filter((k, i, arr) => arr.indexOf(k) === i)
    .sort();
}

function placeholders(s: string): string[] {
  return (s.match(/%\{\w+\}/g) ?? []).sort();
}

const locales = { he, ar, ru, am } as const;
const enFlat = flatten(en as unknown as Tree);
const enKeys = structuralKeys(en as unknown as Tree);

describe.each(Object.entries(locales))('%s locale', (_name, locale) => {
  const flat = flatten(locale as unknown as Tree);

  it('has exactly the keys English has', () => {
    expect(structuralKeys(locale as unknown as Tree)).toEqual(enKeys);
  });

  it('has every plural form English has', () => {
    for (const key of Object.keys(enFlat)) {
      expect(flat[key]).toBeDefined();
    }
  });

  it('keeps the same placeholders as English (count may be dropped in singular forms)', () => {
    for (const [key, value] of Object.entries(flat)) {
      const reference = enFlat[key] ?? enFlat[key.replace(/\.(zero|two|few|many)$/, '.other')];
      if (!reference) continue;
      const want = placeholders(reference).filter((p) => p !== '%{count}');
      const have = placeholders(value).filter((p) => p !== '%{count}');
      expect({ key, have }).toEqual({ key, have: want });
    }
  });

  it('has no empty strings', () => {
    for (const value of Object.values(flat)) expect(value.trim().length).toBeGreaterThan(0);
  });
});
