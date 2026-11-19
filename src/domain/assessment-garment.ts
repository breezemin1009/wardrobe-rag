import type { Garment } from './model.ts';
import { designCategory, emptyView, type DesignProfile } from './design.ts';

const exampleKinds: Record<NonNullable<Garment['demoAsset']>, NonNullable<DesignProfile['garmentType']>> = {
  'white-shirt':'shirt', 'cream-tee':'tshirt', 'blue-shirt':'shirt', 'black-knit':'knit',
  'grey-sweatshirt':'knit', 'olive-shirt':'shirt', 'black-pants':'trousers', denim:'jeans',
  'beige-pants':'trousers', 'blue-stripe':'shirt', 'breton-tee':'tshirt', 'olive-check':'shirt',
  'beige-knit':'knit', 'cream-knit':'knit', 'navy-polo':'tshirt', 'grey-pants':'trousers',
  'olive-pants':'trousers', 'cream-pants':'jeans',
};

/** Old built-in pictures have a known type even if the user renames them.
 * Uploaded legacy photos require explicit review; their names are never a type.
 */
export function assessmentDesign(g: Garment): DesignProfile | undefined {
  if (g.design?.garmentType) return g.design;
  const garmentType = !g.imageUri && g.demoAsset ? exampleKinds[g.demoAsset] : undefined;
  if (!garmentType || designCategory(garmentType) !== g.category) return g.design;
  return { version:1, garmentType, front:emptyView(g.color,g.pattern), back:null, backStatus:'unknown' };
}
