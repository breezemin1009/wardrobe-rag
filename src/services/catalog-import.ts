import { api } from './rag';
import type { Garment } from '@/domain/model';
import { emptyView, type DesignProfile } from '@/domain/design';

type Product = { id: string; name: string; image: string; imageHash: string; category: Garment['category']; audience: 'mens' | 'unisex'; baseColour: string; articleType: string };
const colorMap: Record<string, Garment['color']> = { White: '화이트', Black: '블랙', Blue: '블루', 'Navy Blue': '블루', 'Off White': '크림', Beige: '베이지', Khaki: '베이지', Tan: '베이지', Grey: '그레이', 'Grey Melange': '그레이', Olive: '올리브' };
const kindMap: Record<string, DesignProfile['garmentType']> = { Shirts: 'shirt', Tshirts: 'tshirt', Jeans: 'jeans', Trousers: 'trousers', 'Track Pants': 'trousers', Shorts: 'shorts', Jackets: 'jacket', Blazers: 'jacket', Sweaters: 'knit', Sweatshirts: 'unknown' };

export async function importCatalogProduct(id: string, signal: AbortSignal): Promise<Garment> {
  const p = await api<Product>('/menswear/items/' + encodeURIComponent(id), undefined, signal);
  const color = colorMap[p.baseColour] || '기타';
  return { id: '', name: p.name.slice(0, 60), category: p.category, color, pattern: '기타', occasion: '데일리',
    role: 'candidate', audience: p.audience, sourceProductId: p.id, imageUri: p.image, imageHash: p.imageHash,
    isExample: false, createdAt: Date.now(), design: { version: 1, garmentType: kindMap[p.articleType] || 'unknown',
      front: { ...emptyView(color, '기타'), uncertain: ['baseColor', 'pattern', 'patternColors', 'patternScale', 'contrast', 'accents'] },
      back: null, backStatus: 'unknown' } };
}
