export enum AppSection {
  HOME = 'home',
  FEATURES = 'features',
  DEMO = 'demo',
  PRICING = 'pricing',
}

export interface PricePoint {
  month: string;
  price: number;
}

export interface AnalysisResult {
  productName: string;
  category: string;
  confidenceScore: number;
  matchStatus: 'Exact Match' | 'Similar Variant' | 'New Item';
  suggestedPrice: number;
  priceHistory: PricePoint[];
  inventoryAction: string;
}

export interface PricingTier {
  name: string;
  price: string;
  description: string;
  features: string[];
  cta: string;
  highlight?: boolean;
}