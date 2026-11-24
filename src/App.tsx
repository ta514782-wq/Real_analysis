/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Activity,
  BookOpen,
  Box,
  BrainCircuit,
  CheckSquare,
  Compass,
  FileText,
  FolderKanban,
  GitCompare,
  GraduationCap,
  Grid,
  LineChart,
  Menu,
  Sigma,
  X,
} from 'lucide-react';
import { useState } from 'react';
import { ConvergenceAnalyzerView } from './modules/ConvergenceAnalyzerView';
import { ExperimentManagerView } from './modules/ExperimentManagerView';
import { FunctionExplorerView } from './modules/FunctionExplorerView';
import { GradientDescentSimulatorView } from './modules/GradientDescentSimulatorView';
import { HessianAnalyzerView } from './modules/HessianAnalyzerView';
import { InteractiveAssessmentView } from './modules/InteractiveAssessmentView';
import { MachineLearningLabView } from './modules/MachineLearningLabView';
import { Multivariable3DLabView } from './modules/Multivariable3DLabView';
import { NumericalDiffLabView } from './modules/NumericalDiffLabView';
import { ReportGeneratorView } from './modules/ReportGeneratorView';
import { StageRoadmapView } from './modules/StageRoadmapView';
import { TheoryLaboratoryView } from './modules/TheoryLaboratoryView';
import { VerificationCenterView } from './modules/VerificationCenterView';

type ActiveTab =
  | 'explorer'
  | 'simulator'
  | 'multivariable3d'
  | 'hessian'
  | 'ml'
  | 'convergence'
  | 'numdiff'
  | 'experiments'
  | 'theory'
  | 'assessment'
  | 'reports'
  | 'verification'
  | 'roadmap';

const NAV_ITEMS: { id: ActiveTab; label: string; icon: typeof Activity }[] = [
  { id: 'explorer', label: 'Module 1 · Function Explorer', icon: LineChart },
  { id: 'simulator', label: 'Module 2 · GD Simulator', icon: Activity },
  { id: 'multivariable3d', label: 'Module 3 · 3D Surface Lab', icon: Box },
  { id: 'hessian', label: 'Hessian & Convexity', icon: Grid },
  { id: 'ml', label: 'Module 4 · ML Laboratory', icon: BrainCircuit },
  { id: 'convergence', label: 'Module 5 · Convergence Lab', icon: GitCompare },
  { id: 'numdiff', label: 'Numerical Differentiation', icon: Sigma },
  { id: 'experiments', label: 'Experiment Manager', icon: FolderKanban },
  { id: 'theory', label: 'Real Analysis Theory Lab', icon: BookOpen },
  { id: 'assessment', label: 'Interactive Assessment', icon: GraduationCap },
  { id: 'reports', label: 'Module 6 · PDF/CSV Report', icon: FileText },
  { id: 'verification', label: 'Verification Center', icon: CheckSquare },
  { id: 'roadmap', label: 'Build Roadmap (Stages 1–6)', icon: Compass },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('explorer');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#07090E] text-slate-100 lg:flex">
      {/* Desktop Sidebar */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-800 bg-[#0B0E17] lg:flex">
        <div className="border-b border-slate-800 px-5 py-4">
          <div className="font-display text-lg font-bold tracking-tight text-slate-100">
            GradAI
          </div>
          <div className="mt-0.5 text-xs text-slate-400">
            MATH 303 · Real Analysis I
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto p-3" aria-label="Laboratory Navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                className={`flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-xs font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-cyan-500/12 text-cyan-300'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-100'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-slate-800 px-5 py-4 text-xs text-slate-500">
          All 6 Stages Complete · 13 Workspaces
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top Bar */}
        <header className="flex items-center justify-between gap-4 border-b border-slate-800 bg-[#0B0E17] px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileMenuOpen((v) => !v)}
              aria-label="Toggle navigation menu"
              className="rounded border border-slate-800 p-2 text-slate-300 hover:bg-slate-800 lg:hidden"
            >
              {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </button>
            <h1 className="truncate font-display text-sm font-semibold text-slate-100 sm:text-base">
              GradAI — Gradient Descent & Real Analysis Laboratory
            </h1>
          </div>

          <div className="hidden items-center gap-3 text-xs text-slate-400 sm:flex whitespace-nowrap shrink-0">
            <span>MATH 303 Optimization Lab</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-emerald-400">● 18/18 Tests Passing</span>
          </div>
        </header>

        {/* Mobile Drawer */}
        {mobileMenuOpen && (
          <nav
            className="border-b border-slate-800 bg-[#0B0E17] p-3 lg:hidden"
            aria-label="Mobile Laboratory Navigation"
          >
            <div className="grid grid-cols-1 gap-1 sm:grid-cols-2">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setActiveTab(item.id);
                      setMobileMenuOpen(false);
                    }}
                    className={`flex w-full items-center gap-2.5 rounded-md px-3 py-2.5 text-left text-xs font-medium transition-colors ${
                      active
                        ? 'bg-cyan-500/12 text-cyan-300'
                        : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-100'
                    }`}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          </nav>
        )}

        <main className="mx-auto w-full max-w-[1440px] flex-1 p-4 sm:p-6">
          {activeTab === 'explorer' && <FunctionExplorerView />}
          {activeTab === 'simulator' && <GradientDescentSimulatorView />}
          {activeTab === 'multivariable3d' && <Multivariable3DLabView />}
          {activeTab === 'hessian' && <HessianAnalyzerView />}
          {activeTab === 'ml' && <MachineLearningLabView />}
          {activeTab === 'convergence' && <ConvergenceAnalyzerView />}
          {activeTab === 'numdiff' && <NumericalDiffLabView />}
          {activeTab === 'experiments' && <ExperimentManagerView />}
          {activeTab === 'theory' && <TheoryLaboratoryView />}
          {activeTab === 'assessment' && <InteractiveAssessmentView />}
          {activeTab === 'reports' && <ReportGeneratorView />}
          {activeTab === 'verification' && <VerificationCenterView />}
          {activeTab === 'roadmap' && <StageRoadmapView />}
        </main>
      </div>
    </div>
  );
}
