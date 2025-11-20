import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data url prefix (e.g. "data:image/jpeg;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export const analyzeProductImage = async (file: File): Promise<AnalysisResult> => {
  try {
    const base64Data = await fileToBase64(file);

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: file.type,
              data: base64Data
            }
          },
          {
            text: `You are the core engine of "CatalogSync AI", a software for business inventory matching. 
            
            Analyze this uploaded product image. 
            1. Identify the product.
            2. Simulate a "search" against an existing legacy catalog of 1000+ items.
            3. Determine if this uploaded image matches an item in the old catalog (Exact Match), is a variant (Similar Variant), or is completely new (New Item).
            4. Assign a business category.
            5. Suggest a price and generate a mock 6-month price history (analytics breakdown).
            6. Recommend an inventory action (e.g., "Link to SKU #123", "Create New Record").

            Return strictly valid JSON matching the schema.`
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            productName: { type: Type.STRING },
            category: { type: Type.STRING },
            confidenceScore: { type: Type.NUMBER },
            matchStatus: { type: Type.STRING, enum: ['Exact Match', 'Similar Variant', 'New Item'] },
            suggestedPrice: { type: Type.NUMBER },
            inventoryAction: { type: Type.STRING, description: "Action recommendation based on match status" },
            priceHistory: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  month: { type: Type.STRING },
                  price: { type: Type.NUMBER }
                }
              }
            }
          }
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");

    return JSON.parse(text) as AnalysisResult;

  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    // Fallback mock data in case of API failure or error to ensure demo doesn't crash
    return {
      productName: "Product Analysis Failed",
      category: "Unknown",
      confidenceScore: 0,
      matchStatus: "New Item",
      suggestedPrice: 0,
      priceHistory: [],
      inventoryAction: "Manual Review Required"
    };
  }
};