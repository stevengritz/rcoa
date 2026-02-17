#!/usr/bin/env node
// ============================================================
// RCOA Metrics Verification Suite
// Validates that RCOA implementation matches paper claims
// Run: node verify-metrics.js [--quick]
// ============================================================

"use strict";

const args = process.argv.slice(2);
const QUICK = args.includes("--quick");

// === Math Utilities ===
function rand(a, b) { return a + Math.random() * (b - a); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function gaussRand() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function gammaFn(z) {
  if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gammaFn(1 - z));
  z -= 1;
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}
function levyStep(lam = 1.5) {
  const sU = Math.pow(
    (gammaFn(1 + lam) * Math.sin(Math.PI * lam / 2)) /
    (gammaFn((1 + lam) / 2) * lam * Math.pow(2, (lam - 1) / 2)), 1 / lam);
  const u = gaussRand() * sU, v = Math.abs(gaussRand());
  return u / Math.pow(v, 1 / lam);
}
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}
function weibullDeg(shape, scale) {
  const u = Math.random();
  return scale * Math.pow(-Math.log(1 - u), 1 / shape);
}

// Wilcoxon signed-rank test (approximate p-value)
function wilcoxonSignedRank(a, b) {
  const diffs = a.map((v, i) => v - b[i]).filter(d => d !== 0);
  const absDiffs = diffs.map((d, i) => ({ d, abs: Math.abs(d), idx: i }));
  absDiffs.sort((x, y) => x.abs - y.abs);
  
  let rank = 1;
  absDiffs.forEach((item, i) => {
    item.rank = rank;
    rank++;
  });
  
  let Wplus = 0, Wminus = 0;
  absDiffs.forEach(item => {
    if (diffs[item.idx] > 0) Wplus += item.rank;
    else Wminus += item.rank;
  });
  
  const n = diffs.length;
  const W = Math.min(Wplus, Wminus);
  const mu = n * (n + 1) / 4;
  const sigma = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24);
  const z = (W - mu) / sigma;
  
  // Approximate two-tailed p-value
  const p = 2 * (1 - 0.5 * (1 + erf(Math.abs(z) / Math.sqrt(2))));
  return { W, z, p, Wplus, Wminus };
}

function erf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

// === Paper Target Values (from RCOA_Revised_Paper.md) ===
const TARGETS = {
  smp: {
    rcoa: { mean: 0.889, std: 0.019 },
    pso: { mean: 0.861, std: 0.028 },
    ga: { mean: 0.847, std: 0.031 },
    description: "SMP 20-component reliability (Section 8.3)"
  },
  featureSelection: {
    rcoa: { acc: 0.816, noiseDims: 4.2 },
    ga: { acc: 0.723 },
    pso: { acc: 0.687 },
    description: "Feature Selection D=12, 5 noise dims (Section 9.2)"
  }
};

// === SMP Benchmark ===
function runSMPVerification(runs = 51) {
  console.log("\n" + "=".repeat(70));
  console.log("  SMP VERIFICATION (Paper Target: RCOA 88.9% vs PSO 86.1%)");
  console.log("=".repeat(70));
  
  const N = 20, M = 6, iterations = 500;
  const gamma = 0.5, eta = 0.25, alpha = 0.2, tauStag = 6, tauMolt = 15;
  const severity = 5;
  
  const rcoaResults = [], psoResults = [], gaResults = [];
  
  for (let run = 0; run < runs; run++) {
    process.stdout.write(`\r  Running... ${run + 1}/${runs}`);
    
    // Generate shared component profiles
    const components = Array.from({ length: N }, () => ({
      shape: rand(1.5, 3.0),
      scale: rand(0.02, 0.06),
    }));
    
    // === RCOA ===
    const rcoaHealth = Array(N).fill(1.0);
    const stagnation = Array(N).fill(0);
    const lastHealth = Array(N).fill(1.0);
    const locked = Array(N).fill(false);
    const lockTimers = Array(N).fill(0);
    const density = Array(N).fill(0);
    const crabTargets = Array(M).fill(0);
    const crabImprovement = Array(M).fill(0);
    const crabMolting = Array(M).fill(false);
    const crabMoltTimer = Array(M).fill(0);
    
    for (let t = 0; t < iterations; t++) {
      // Degradation
      for (let i = 0; i < N; i++) {
        if (!locked[i]) {
          const deg = weibullDeg(components[i].shape, components[i].scale) * (severity / 8);
          rcoaHealth[i] = clamp(rcoaHealth[i] - deg, 0, 1);
        }
        if (lockTimers[i] > 0) {
          lockTimers[i]--;
          if (lockTimers[i] <= 0) locked[i] = false;
        }
      }
      
      // Density-penalized assignment (instant symbiosis)
      density.fill(0);
      for (let j = 0; j < M; j++) {
        if (crabMolting[j]) {
          crabMoltTimer[j]--;
          if (crabMoltTimer[j] <= 0) {
            crabMolting[j] = false;
            crabImprovement[j] = 0;
          }
          continue;
        }
        
        let bestAttr = -Infinity, bestIdx = 0;
        for (let i = 0; i < N; i++) {
          const urgency = (1 - rcoaHealth[i]) * 2 + 0.5;
          const attr = urgency / (1 + gamma * density[i]);
          if (attr > bestAttr) { bestAttr = attr; bestIdx = i; }
        }
        crabTargets[j] = bestIdx;
        density[bestIdx]++;
      }
      
      // Apply operators
      for (let j = 0; j < M; j++) {
        if (crabMolting[j]) continue;
        const idx = crabTargets[j];
        if (locked[idx]) continue;
        
        // Fertilization
        const densF = 1 / (1 + gamma * density[idx]);
        const repair = eta * densF * (1 - rcoaHealth[idx]);
        rcoaHealth[idx] = clamp(rcoaHealth[idx] + repair, 0, 1);
        crabImprovement[j] += repair;
        
        // Bioturbation
        if (Math.abs(rcoaHealth[idx] - lastHealth[idx]) < 0.005) {
          stagnation[idx]++;
        } else {
          stagnation[idx] = 0;
        }
        lastHealth[idx] = rcoaHealth[idx];
        
        if (stagnation[idx] >= tauStag) {
          const jolt = Math.abs(levyStep(1.5)) * alpha * 0.08;
          rcoaHealth[idx] = clamp(rcoaHealth[idx] + jolt, 0, 1);
          stagnation[idx] = 0;
        }
      }
      
      // Molting
      if (t % tauMolt === 0 && t > 0) {
        const sortedCrabs = [];
        for (let j = 0; j < M; j++) {
          if (!crabMolting[j]) sortedCrabs.push({ j, imp: crabImprovement[j] });
        }
        sortedCrabs.sort((a, b) => a.imp - b.imp);
        const moltN = Math.max(1, Math.floor(sortedCrabs.length * 0.1));
        const sortedHealth = rcoaHealth.map((h, i) => ({ i, h })).sort((a, b) => b.h - a.h);
        
        for (let k = 0; k < moltN && k < sortedCrabs.length; k++) {
          crabMolting[sortedCrabs[k].j] = true;
          crabMoltTimer[sortedCrabs[k].j] = 5;
          locked[sortedHealth[k].i] = true;
          lockTimers[sortedHealth[k].i] = 5;
        }
      }
    }
    rcoaResults.push(mean(rcoaHealth));
    
    // === PSO ===
    const psoHealth = Array(N).fill(1.0);
    const psoTargets = Array.from({ length: M }, () => Math.floor(Math.random() * N));
    let gBest = 0;
    
    for (let t = 0; t < iterations; t++) {
      for (let i = 0; i < N; i++) {
        const deg = weibullDeg(components[i].shape, components[i].scale) * (severity / 8);
        psoHealth[i] = clamp(psoHealth[i] - deg, 0, 1);
      }
      
      for (let j = 0; j < M; j++) {
        psoHealth[psoTargets[j]] = clamp(psoHealth[psoTargets[j]] + 0.15, 0, 1);
        if (psoHealth[psoTargets[j]] > psoHealth[gBest]) gBest = psoTargets[j];
        const r2 = Math.random();
        psoTargets[j] = clamp(Math.round(0.5 * psoTargets[j] + 0.5 * r2 * (gBest - psoTargets[j])), 0, N - 1);
      }
    }
    psoResults.push(mean(psoHealth));
    
    // === GA ===
    const gaHealth = Array(N).fill(1.0);
    let gaPop = Array.from({ length: 20 }, () =>
      Array.from({ length: N }, () => Math.floor(Math.random() * 4))
    );
    
    for (let t = 0; t < iterations; t++) {
      for (let i = 0; i < N; i++) {
        const deg = weibullDeg(components[i].shape, components[i].scale) * (severity / 8);
        gaHealth[i] = clamp(gaHealth[i] - deg, 0, 1);
      }
      
      let bestPlan = gaPop[0], bestScore = -Infinity;
      for (const plan of gaPop) {
        let cost = 0, score = 0;
        for (let i = 0; i < N; i++) {
          const repair = [0, 0.05, 0.15, 0.3][plan[i]];
          const repairCost = [0, 1, 3, 7][plan[i]];
          cost += repairCost;
          if (cost <= 40) score += gaHealth[i] + repair;
        }
        if (score > bestScore) { bestScore = score; bestPlan = plan; }
      }
      
      let crewsUsed = 0;
      for (let i = 0; i < N && crewsUsed < M; i++) {
        if (bestPlan[i] > 0) {
          gaHealth[i] = clamp(gaHealth[i] + [0, 0.05, 0.15, 0.3][bestPlan[i]], 0, 1);
          crewsUsed++;
        }
      }
      
      for (let p = 0; p < gaPop.length; p++) {
        if (Math.random() < 0.3) {
          const donor = gaPop[Math.floor(Math.random() * gaPop.length)];
          gaPop[p] = gaPop[p].map((v, i) => Math.random() < 0.5 ? v : donor[i]);
        }
        gaPop[p] = gaPop[p].map(v => Math.random() < 0.05 ? Math.floor(Math.random() * 4) : v);
      }
    }
    gaResults.push(mean(gaHealth));
  }
  
  process.stdout.write("\r" + " ".repeat(40) + "\r");
  
  // Results
  console.log("\n  Results (avg component health):");
  console.log(`    RCOA:  ${(mean(rcoaResults) * 100).toFixed(2)}% ± ${(std(rcoaResults) * 100).toFixed(2)}%`);
  console.log(`    PSO:   ${(mean(psoResults) * 100).toFixed(2)}% ± ${(std(psoResults) * 100).toFixed(2)}%`);
  console.log(`    GA:    ${(mean(gaResults) * 100).toFixed(2)}% ± ${(std(gaResults) * 100).toFixed(2)}%`);
  
  // Paper targets
  console.log("\n  Paper Targets:");
  console.log(`    RCOA:  ${(TARGETS.smp.rcoa.mean * 100).toFixed(1)}% ± ${(TARGETS.smp.rcoa.std * 100).toFixed(1)}%`);
  console.log(`    PSO:   ${(TARGETS.smp.pso.mean * 100).toFixed(1)}% ± ${(TARGETS.smp.pso.std * 100).toFixed(1)}%`);
  
  // Win counts
  const rcoaWinsPSO = rcoaResults.filter((v, i) => v > psoResults[i]).length;
  const rcoaWinsGA = rcoaResults.filter((v, i) => v > gaResults[i]).length;
  console.log(`\n  Win counts (${runs} runs):`);
  console.log(`    RCOA beats PSO: ${rcoaWinsPSO}/${runs} (${(rcoaWinsPSO/runs*100).toFixed(1)}%)`);
  console.log(`    RCOA beats GA:  ${rcoaWinsGA}/${runs} (${(rcoaWinsGA/runs*100).toFixed(1)}%)`);
  
  // Statistical test
  const wPSO = wilcoxonSignedRank(rcoaResults, psoResults);
  const wGA = wilcoxonSignedRank(rcoaResults, gaResults);
  console.log("\n  Wilcoxon Signed-Rank Test:");
  console.log(`    RCOA vs PSO: W=${wPSO.W.toFixed(0)}, z=${wPSO.z.toFixed(3)}, p=${wPSO.p.toFixed(4)} ${wPSO.p < 0.05 ? "✓ SIGNIFICANT" : "✗ not significant"}`);
  console.log(`    RCOA vs GA:  W=${wGA.W.toFixed(0)}, z=${wGA.z.toFixed(3)}, p=${wGA.p.toFixed(4)} ${wGA.p < 0.05 ? "✓ SIGNIFICANT" : "✗ not significant"}`);
  
  // Validation
  const pass = rcoaWinsPSO / runs >= 0.7;
  console.log(`\n  ${pass ? "✓ PASS" : "✗ FAIL"}: RCOA beats PSO ≥70% of runs (target: 70%, actual: ${(rcoaWinsPSO/runs*100).toFixed(1)}%)`);
  
  return { rcoaResults, psoResults, gaResults, pass };
}

// === Feature Selection Benchmark ===
function runFeatureSelectionVerification(runs = 51) {
  console.log("\n" + "=".repeat(70));
  console.log("  FEATURE SELECTION VERIFICATION (Paper Target: RCOA 81.6% vs GA 72.3%)");
  console.log("=".repeat(70));
  
  const D_total = 12, D_noise = 5, D_signal = D_total - D_noise;
  const nSignal = 80, nNoise = 120, nClusters = 3;
  const iterations = 200;
  const N = 8; // Rice agents
  const epsilon = 0.15;
  
  const rcoaAccs = [], gaAccs = [], psoAccs = [];
  const rcoaWeeded = [];
  
  for (let run = 0; run < runs; run++) {
    process.stdout.write(`\r  Running... ${run + 1}/${runs}`);
    
    // Generate dataset
    const centers = Array.from({ length: nClusters }, () =>
      Array.from({ length: D_total }, (_, d) => d < D_signal ? rand(-2, 2) : 0)
    );
    
    const points = [];
    for (let c = 0; c < nClusters; c++) {
      const perC = Math.floor(nSignal / nClusters);
      for (let i = 0; i < perC; i++) {
        const feats = centers[c].map((v, d) => d < D_signal ? v + gaussRand() * 0.5 : rand(-3, 3));
        points.push({ features: feats, isSignal: true });
      }
    }
    for (let i = 0; i < nNoise; i++) {
      points.push({ features: Array.from({ length: D_total }, () => rand(-4, 4)), isSignal: false });
    }
    
    const isSignalDim = Array.from({ length: D_total }, (_, d) => d < D_signal);
    
    function fitness(mask, centroid) {
      let correct = 0;
      for (const p of points) {
        let d = 0;
        for (let k = 0; k < D_total; k++) d += mask[k] * (p.features[k] - centroid[k]) ** 2;
        d = Math.sqrt(d);
        const classified = d < 2.0;
        if ((p.isSignal && classified) || (!p.isSignal && !classified)) correct++;
      }
      let penalty = 0;
      for (let k = D_signal; k < D_total; k++) penalty += mask[k] * 0.05;
      return correct / points.length - penalty;
    }
    
    // === RCOA with weeding ===
    let bestAcc = 0;
    const rcoaCentroids = Array.from({ length: N }, () => ({
      features: Array.from({ length: D_total }, () => rand(-2, 2)),
      mask: Array(D_total).fill(1),
    }));
    const featureWeights = Array(D_total).fill(1.0);
    
    for (let t = 0; t < iterations; t++) {
      for (const rc of rcoaCentroids) {
        const fit = fitness(rc.mask, rc.features);
        
        // Weeding: test D/3 dimensions
        const dimsToTest = Math.max(1, Math.ceil(D_total / 3));
        for (let k = 0; k < dimsToTest; k++) {
          const d = Math.floor(Math.random() * D_total);
          if (rc.mask[d] === 0) continue;
          const trialMask = rc.mask.slice();
          trialMask[d] = 0;
          const trialFit = fitness(trialMask, rc.features);
          if (trialFit > fit) {
            rc.mask[d] = 0;
            featureWeights[d] = Math.max(featureWeights[d] - 0.12, 0);
          }
        }
        
        // Fertilization
        for (let d = 0; d < D_total; d++) {
          if (rc.mask[d] === 0) continue;
          rc.features[d] += 0.1 * gaussRand() * 0.3;
        }
        
        const newFit = fitness(rc.mask, rc.features);
        if (newFit > bestAcc) bestAcc = newFit;
      }
    }
    
    const noiseDimsWeeded = featureWeights.reduce((s, w, d) => s + (!isSignalDim[d] && w < 0.3 ? 1 : 0), 0);
    rcoaAccs.push(bestAcc);
    rcoaWeeded.push(noiseDimsWeeded);
    
    // === GA ===
    let gaBestAcc = 0;
    let gaPop = Array.from({ length: 16 }, () => ({
      mask: Array.from({ length: D_total }, () => Math.random() > 0.3 ? 1 : 0),
      features: Array.from({ length: D_total }, () => rand(-2, 2)),
    }));
    
    for (let t = 0; t < iterations; t++) {
      const fits = gaPop.map(ind => fitness(ind.mask, ind.features));
      const maxFit = Math.max(...fits);
      if (maxFit > gaBestAcc) gaBestAcc = maxFit;
      
      const newPop = [];
      for (let i = 0; i < gaPop.length; i++) {
        const a = Math.floor(Math.random() * gaPop.length);
        const b = Math.floor(Math.random() * gaPop.length);
        const parent = fits[a] > fits[b] ? gaPop[a] : gaPop[b];
        newPop.push({
          mask: parent.mask.map(v => Math.random() < 0.05 ? (1 - v) : v),
          features: parent.features.map(f => f + gaussRand() * 0.1),
        });
      }
      gaPop = newPop;
    }
    gaAccs.push(gaBestAcc);
    
    // === PSO ===
    let psoBestAcc = 0;
    let psoParticles = Array.from({ length: 16 }, () => ({
      mask: Array.from({ length: D_total }, () => rand(0, 1)),
      features: Array.from({ length: D_total }, () => rand(-2, 2)),
      v: Array.from({ length: D_total }, () => rand(-0.1, 0.1)),
    }));
    
    for (let t = 0; t < iterations; t++) {
      for (const p of psoParticles) {
        const roundedMask = p.mask.map(v => v > 0.5 ? 1 : 0);
        const fit = fitness(roundedMask, p.features);
        if (fit > psoBestAcc) psoBestAcc = fit;
        p.v = p.v.map(v => 0.7 * v + 0.3 * Math.random() * rand(-0.5, 0.5));
        p.mask = p.mask.map((m, d) => clamp(m + p.v[d], 0, 1));
        p.features = p.features.map(f => f + gaussRand() * 0.05);
      }
    }
    psoAccs.push(psoBestAcc);
  }
  
  process.stdout.write("\r" + " ".repeat(40) + "\r");
  
  // Results
  console.log("\n  Results:");
  console.log(`    RCOA:  ${(mean(rcoaAccs) * 100).toFixed(2)}% (avg ${mean(rcoaWeeded).toFixed(1)}/${D_noise} noise dims weeded)`);
  console.log(`    GA:    ${(mean(gaAccs) * 100).toFixed(2)}%`);
  console.log(`    PSO:   ${(mean(psoAccs) * 100).toFixed(2)}%`);
  
  console.log("\n  Paper Targets:");
  console.log(`    RCOA:  ${(TARGETS.featureSelection.rcoa.acc * 100).toFixed(1)}% (${TARGETS.featureSelection.rcoa.noiseDims}/${D_noise} noise dims)`);
  console.log(`    GA:    ${(TARGETS.featureSelection.ga.acc * 100).toFixed(1)}%`);
  
  // Win counts
  const rcoaWinsGA = rcoaAccs.filter((v, i) => v > gaAccs[i]).length;
  const rcoaWinsPSO = rcoaAccs.filter((v, i) => v > psoAccs[i]).length;
  console.log(`\n  Win counts (${runs} runs):`);
  console.log(`    RCOA beats GA:  ${rcoaWinsGA}/${runs} (${(rcoaWinsGA/runs*100).toFixed(1)}%)`);
  console.log(`    RCOA beats PSO: ${rcoaWinsPSO}/${runs} (${(rcoaWinsPSO/runs*100).toFixed(1)}%)`);
  
  // Statistical test
  const wGA = wilcoxonSignedRank(rcoaAccs, gaAccs);
  console.log("\n  Wilcoxon Signed-Rank Test:");
  console.log(`    RCOA vs GA: W=${wGA.W.toFixed(0)}, z=${wGA.z.toFixed(3)}, p=${wGA.p.toFixed(4)} ${wGA.p < 0.05 ? "✓ SIGNIFICANT" : "✗ not significant"}`);
  
  // Validation
  const pass = rcoaWinsGA / runs >= 0.8;
  console.log(`\n  ${pass ? "✓ PASS" : "✗ FAIL"}: RCOA beats GA ≥80% of runs (target: 80%, actual: ${(rcoaWinsGA/runs*100).toFixed(1)}%)`);
  
  return { rcoaAccs, gaAccs, psoAccs, pass };
}

// === Main ===
function main() {
  console.log("╔══════════════════════════════════════════════════════════════════╗");
  console.log("║  RCOA METRICS VERIFICATION SUITE                                 ║");
  console.log("║  Validates implementation against paper claims                    ║");
  console.log("╚══════════════════════════════════════════════════════════════════╝");
  
  const runs = QUICK ? 11 : 51;
  console.log(`\n  Running with ${runs} independent trials ${QUICK ? "(--quick mode)" : "(paper standard)"}`);
  
  const smpResult = runSMPVerification(runs);
  const fsResult = runFeatureSelectionVerification(runs);
  
  console.log("\n" + "=".repeat(70));
  console.log("  VERIFICATION SUMMARY");
  console.log("=".repeat(70));
  console.log(`  SMP Benchmark:              ${smpResult.pass ? "✓ PASS" : "✗ FAIL"}`);
  console.log(`  Feature Selection:          ${fsResult.pass ? "✓ PASS" : "✗ FAIL"}`);
  console.log("=".repeat(70));
  
  const allPass = smpResult.pass && fsResult.pass;
  console.log(`\n  Overall: ${allPass ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED"}`);
  console.log("  Run with --quick for faster execution (11 runs instead of 51)");
  
  process.exit(allPass ? 0 : 1);
}

main();
