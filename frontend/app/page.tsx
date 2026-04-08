import { Suspense } from "react";
import { promises as fs } from "fs";
import path from "path";
import { PredictionsData } from "@/types/predictions";
import PredictionsDashboard from "@/components/PredictionsDashboard";

async function loadPredictions(): Promise<PredictionsData> {
  // Read static predictions file committed to the repo.
  // Vercel redeploys on every push (triggered by the GitHub Actions daily job),
  // so this file is always fresh without needing runtime caching.
  const filePath = path.join(process.cwd(), "public", "data", "predictions.json");
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw) as PredictionsData;
}

export default async function HomePage() {
  const data = await loadPredictions();

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="max-w-4xl mx-auto px-4 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900">Football ML Predictions</h1>
          <p className="text-slate-500 mt-1 text-sm">
            Ensemble model (Poisson + XGBoost + LightGBM) · Last updated{" "}
            {new Date(data.generated_at).toLocaleString("en-GB", {
              day: "numeric",
              month: "short",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
              timeZoneName: "short",
            })}
          </p>
          <p className="mt-2">
            <a href="/methodology" className="text-sm text-blue-600 hover:text-blue-800 underline">
              How this works →
            </a>
          </p>
        </div>

        <Suspense fallback={<div className="text-slate-400">Loading predictions…</div>}>
          <PredictionsDashboard data={data} />
        </Suspense>
      </div>
    </main>
  );
}
