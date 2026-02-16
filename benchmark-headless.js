#!/usr/bin/env node
// ============================================================
// RCOA Headless Benchmark
// Runs RCOA against PSO, DE, and GA on standard benchmark
// functions, plus an SMP instance and feature selection task.
// No browser required — pure Node.js.
// ============================================================

"use strict";

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
  z -= 1; const g = 7;
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

// === Benchmark Functions (minimization) ===
const benchmarks = {
  sphere: {
    name: "Sphere (Unimodal)",
    fn: (x) => x.reduce((s, v) => s + v * v, 0),
    bounds: [-100, 100],
    optimum: 0,
  },
  rastrigin: {
    name: "Rastrigin (Multimodal)",
    fn: (x) => 10 * x.length + x.reduce((s, v) => s + v * v - 10 * Math.cos(2 * Math.PI * v), 0),
    bounds: [-5.12, 5.12],
    optimum: 0,
  },
  ackley: {
    name: "Ackley (Multimodal)",
    fn: (x) => {
      const D = x.length;
      const a = 20, b = 0.2, c = 2 * Math.PI;
      const s1 = x.reduce((s, v) => s + v * v, 0) / D;
      const s2 = x.reduce((s, v) => s + Math.cos(c * v), 0) / D;
      return -a * Math.exp(-b * Math.sqrt(s1)) - Math.exp(s2) + a + Math.E;
    },
    bounds: [-32, 32],
    optimum: 0,
  },
  rosenbrock: {
    name: "Rosenbrock (Valley)",
    fn: (x) => {
      let s = 0;
      for (let i = 0; i < x.length - 1; i++)
        s += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (x[i] - 1) ** 2;
      return s;
    },
    bounds: [-30, 30],
    optimum: 0,
  },
  griewank: {
    name: "Griewank (Multimodal)",
    fn: (x) => {
      const sum = x.reduce((s, v) => s + v * v / 4000, 0);
      const prod = x.reduce((p, v, i) => p * Math.cos(v / Math.sqrt(i + 1)), 1);
      return sum - prod + 1;
    },
    bounds: [-600, 600],
    optimum: 0,
  },
  noisy_sphere: {
    name: "Noisy Sphere (D/2 noise dims)",
    fn: (x) => {
      // Only first half dimensions matter; second half is noise
      const D = x.length;
      const Dsig = Math.ceil(D / 2);
      let s = 0;
      for (let i = 0; i < Dsig; i++) s += x[i] * x[i];
      // Noise dimensions add random penalty if not pruned
      for (let i = Dsig; i < D; i++) s += Math.abs(x[i]) * 0.01;
      return s;
    },
    bounds: [-100, 100],
    optimum: 0,
    hasNoisyDims: true,
  },
};

// === RCOA Implementation (Headless) ===
function runRCOA(fn, D, bounds, maxFEs, params = {}) {
  const N = params.N || 20;                    // Rice agents
  const M = params.M || 15;                    // Crab agents
  const gamma = params.gamma || 0.5;           // Cannibalism
  const eta = params.eta || 0.25;              // Fertilization
  const alpha = params.alpha || 0.2;           // Bioturbation
  const tauStag = params.tauStag || 6;         // Stagnation threshold
  const tauMolt = params.tauMolt || 15;        // Molting period
  const epsilon = params.epsilon || 0.15;      // Weeding threshold
  const weedEnabled = params.weedEnabled !== false;
  const [lb, ub] = [bounds[0], bounds[1]];
  let FEs = 0;

  function evalFn(x) { FEs++; return fn(x); }

  // Initialize Rice (LHS-like)
  const rice = [];
  for (let i = 0; i < N; i++) {
    const x = Array.from({ length: D }, () => rand(lb, ub));
    const fit = evalFn(x);
    rice.push({
      x, fitness: fit, mask: Array(D).fill(1),
      stagnation: 0, lastFitness: fit,
      locked: false, lockTimer: 0, crabDensity: 0,
    });
  }

  // Initialize Crabs
  const crabs = [];
  for (let j = 0; j < M; j++) {
    const host = rice[j % N];
    const x = host.x.map(v => v + gaussRand() * (ub - lb) * 0.05);
    crabs.push({
      x, vx: Array(D).fill(0),
      pBest: host.x.slice(), pBestFit: host.fitness,
      state: 'Foraging', target: null,
      recentImprovement: 0, moltTimer: 0,
    });
  }

  let gBest = rice.reduce((best, r) => r.fitness < best.fitness ? r : best, rice[0]);
  let gBestX = gBest.x.slice();
  let gBestFit = gBest.fitness;
  let iter = 0;
  const convergence = [];

  while (FEs < maxFEs) {
    iter++;

    // Reset density
    rice.forEach(r => { r.crabDensity = 0; });
    crabs.forEach(c => {
      if (c.state === 'Symbiosis' && c.target !== null) rice[c.target].crabDensity++;
    });

    // Update locks
    rice.forEach(r => {
      if (r.lockTimer > 0) { r.lockTimer--; if (r.lockTimer <= 0) r.locked = false; }
    });

    // Crab loop
    for (const c of crabs) {
      if (FEs >= maxFEs) break;

      if (c.state === 'Molting') {
        c.moltTimer--;
        if (c.moltTimer <= 0) { c.state = 'Foraging'; }
        continue;
      }

      // Target selection: density-penalized attractiveness (lower fitness = more attractive for minimization)
      let bestAttr = -Infinity, bestIdx = 0;
      rice.forEach((r, i) => {
        const urgency = r.fitness + 1; // higher fitness = worse = more urgent
        const attr = urgency / (1 + gamma * r.crabDensity);
        if (attr > bestAttr) { bestAttr = attr; bestIdx = i; }
      });
      c.target = bestIdx;
      const target = rice[bestIdx];

      // Symbiosis: apply operators
      c.state = 'Symbiosis';

      // Operator A: Weeding (stochastic - test D/3 random dims)
      if (weedEnabled && !target.locked && D > 2) {
        const dimsToTest = Math.max(1, Math.ceil(D / 3));
        for (let k = 0; k < dimsToTest; k++) {
          if (FEs >= maxFEs) break;
          const d = Math.floor(Math.random() * D);
          if (target.mask[d] === 0) continue;

          const trial = target.x.slice();
          trial[d] = 0;
          const trialFit = evalFn(trial);
          const sensitivity = target.fitness - trialFit; // positive = dim helps (for min, negative means removing helps)

          if (sensitivity > -epsilon) { // dimension doesn't help enough
            // Check if removing it actually improves
            if (trialFit < target.fitness) {
              target.x[d] = 0;
              target.mask[d] = 0;
              target.fitness = trialFit;
            }
          }
        }
      }

      // Operator B: Fertilization
      if (!target.locked) {
        const densF = 1 / (1 + gamma * target.crabDensity);
        const newX = target.x.slice();
        for (let d = 0; d < D; d++) {
          if (target.mask[d] === 0) continue;
          newX[d] += eta * densF * (c.pBest[d] - target.x[d]);
          newX[d] = clamp(newX[d], lb, ub);
        }
        if (FEs < maxFEs) {
          const newFit = evalFn(newX);
          if (newFit < target.fitness) {
            target.x = newX;
            target.fitness = newFit;
            c.recentImprovement += target.fitness - newFit;
          }
          if (newFit < c.pBestFit) {
            c.pBest = newX.slice();
            c.pBestFit = newFit;
          }
        }
      }

      // Operator C: Bioturbation
      if (!target.locked) {
        if (Math.abs(target.fitness - target.lastFitness) < 1e-8) {
          target.stagnation++;
        } else {
          target.stagnation = 0;
        }
        target.lastFitness = target.fitness;

        if (target.stagnation >= tauStag) {
          const newX = target.x.slice();
          for (let d = 0; d < D; d++) {
            if (target.mask[d] === 0) continue;
            newX[d] += alpha * levyStep(1.5) * (ub - lb) * 0.01;
            newX[d] = clamp(newX[d], lb, ub);
          }
          if (FEs < maxFEs) {
            const newFit = evalFn(newX);
            if (newFit < target.fitness) {
              target.x = newX;
              target.fitness = newFit;
            }
          }
          target.stagnation = 0;
        }
      }

      // Update global best
      if (target.fitness < gBestFit) {
        gBestX = target.x.slice();
        gBestFit = target.fitness;
      }
    }

    // Molting
    if (iter % tauMolt === 0) {
      const sortedCrabs = crabs.filter(c => c.state !== 'Molting')
        .sort((a, b) => a.recentImprovement - b.recentImprovement);
      const moltN = Math.max(1, Math.floor(sortedCrabs.length * 0.1));
      const topRice = [...rice].sort((a, b) => a.fitness - b.fitness).slice(0, moltN);

      for (let k = 0; k < moltN && k < sortedCrabs.length; k++) {
        sortedCrabs[k].state = 'Molting';
        sortedCrabs[k].moltTimer = 5;
        sortedCrabs[k].recentImprovement = 0;
        topRice[k % topRice.length].locked = true;
        topRice[k % topRice.length].lockTimer = 5;
      }
    }

    convergence.push(gBestFit);
  }

  return { bestFit: gBestFit, bestX: gBestX, FEs, convergence };
}

// === PSO Implementation ===
function runPSO(fn, D, bounds, maxFEs, popSize = 35) {
  const [lb, ub] = [bounds[0], bounds[1]];
  let FEs = 0;
  function evalFn(x) { FEs++; return fn(x); }

  const omega = 0.729, c1 = 1.49618, c2 = 1.49618;
  const particles = [];
  let gBestX = null, gBestFit = Infinity;

  for (let i = 0; i < popSize; i++) {
    const x = Array.from({ length: D }, () => rand(lb, ub));
    const v = Array.from({ length: D }, () => rand(-(ub - lb) * 0.1, (ub - lb) * 0.1));
    const fit = evalFn(x);
    particles.push({ x, v, pBest: x.slice(), pBestFit: fit });
    if (fit < gBestFit) { gBestFit = fit; gBestX = x.slice(); }
  }

  const convergence = [];

  while (FEs < maxFEs) {
    for (const p of particles) {
      if (FEs >= maxFEs) break;
      for (let d = 0; d < D; d++) {
        const r1 = Math.random(), r2 = Math.random();
        p.v[d] = omega * p.v[d] + c1 * r1 * (p.pBest[d] - p.x[d]) + c2 * r2 * (gBestX[d] - p.x[d]);
        p.x[d] = clamp(p.x[d] + p.v[d], lb, ub);
      }
      const fit = evalFn(p.x);
      if (fit < p.pBestFit) { p.pBestFit = fit; p.pBest = p.x.slice(); }
      if (fit < gBestFit) { gBestFit = fit; gBestX = p.x.slice(); }
    }
    convergence.push(gBestFit);
  }

  return { bestFit: gBestFit, bestX: gBestX, FEs, convergence };
}

// === DE Implementation ===
function runDE(fn, D, bounds, maxFEs, popSize = 35) {
  const [lb, ub] = [bounds[0], bounds[1]];
  const F = 0.5, CR = 0.9;
  let FEs = 0;
  function evalFn(x) { FEs++; return fn(x); }

  let pop = [];
  let fits = [];
  let gBestFit = Infinity, gBestX = null;

  for (let i = 0; i < popSize; i++) {
    const x = Array.from({ length: D }, () => rand(lb, ub));
    const fit = evalFn(x);
    pop.push(x);
    fits.push(fit);
    if (fit < gBestFit) { gBestFit = fit; gBestX = x.slice(); }
  }

  const convergence = [];

  while (FEs < maxFEs) {
    for (let i = 0; i < popSize; i++) {
      if (FEs >= maxFEs) break;
      // Mutation: rand/1
      let a, b, c2;
      do { a = Math.floor(Math.random() * popSize); } while (a === i);
      do { b = Math.floor(Math.random() * popSize); } while (b === i || b === a);
      do { c2 = Math.floor(Math.random() * popSize); } while (c2 === i || c2 === a || c2 === b);

      const jrand = Math.floor(Math.random() * D);
      const trial = pop[i].slice();
      for (let d = 0; d < D; d++) {
        if (Math.random() < CR || d === jrand) {
          trial[d] = clamp(pop[a][d] + F * (pop[b][d] - pop[c2][d]), lb, ub);
        }
      }
      const trialFit = evalFn(trial);
      if (trialFit <= fits[i]) {
        pop[i] = trial;
        fits[i] = trialFit;
        if (trialFit < gBestFit) { gBestFit = trialFit; gBestX = trial.slice(); }
      }
    }
    convergence.push(gBestFit);
  }

  return { bestFit: gBestFit, bestX: gBestX, FEs, convergence };
}

// === GA Implementation ===
function runGA(fn, D, bounds, maxFEs, popSize = 35) {
  const [lb, ub] = [bounds[0], bounds[1]];
  const mutRate = 0.1, mutScale = 0.1;
  let FEs = 0;
  function evalFn(x) { FEs++; return fn(x); }

  let pop = [];
  let fits = [];
  let gBestFit = Infinity, gBestX = null;

  for (let i = 0; i < popSize; i++) {
    const x = Array.from({ length: D }, () => rand(lb, ub));
    const fit = evalFn(x);
    pop.push(x);
    fits.push(fit);
    if (fit < gBestFit) { gBestFit = fit; gBestX = x.slice(); }
  }

  const convergence = [];

  while (FEs < maxFEs) {
    const newPop = [];
    const newFits = [];

    // Elitism: keep best
    const bestIdx = fits.indexOf(Math.min(...fits));
    newPop.push(pop[bestIdx].slice());
    newFits.push(fits[bestIdx]);

    while (newPop.length < popSize) {
      if (FEs >= maxFEs) break;
      // Tournament selection
      const a = Math.floor(Math.random() * popSize);
      const b = Math.floor(Math.random() * popSize);
      const p1 = fits[a] < fits[b] ? pop[a] : pop[b];
      const c2 = Math.floor(Math.random() * popSize);
      const d = Math.floor(Math.random() * popSize);
      const p2 = fits[c2] < fits[d] ? pop[c2] : pop[d];

      // Crossover
      const child = p1.map((v, i) => Math.random() < 0.5 ? v : p2[i]);

      // Mutation
      for (let i = 0; i < D; i++) {
        if (Math.random() < mutRate) {
          child[i] = clamp(child[i] + gaussRand() * (ub - lb) * mutScale, lb, ub);
        }
      }

      const fit = evalFn(child);
      newPop.push(child);
      newFits.push(fit);
      if (fit < gBestFit) { gBestFit = fit; gBestX = child.slice(); }
    }

    pop = newPop;
    fits = newFits;
    convergence.push(gBestFit);
  }

  return { bestFit: gBestFit, bestX: gBestX, FEs, convergence };
}

// === SMP Benchmark ===
function runSMPBenchmark(runs = 21) {
  console.log("\n" + "=".repeat(70));
  console.log("  SELECTIVE MAINTENANCE PROBLEM (20 components, 6 crews)");
  console.log("=".repeat(70));

  const N = 20; // components
  const maxIters = 500;
  const results = { rcoa: [], pso: [], ga: [] };

  for (let run = 0; run < runs; run++) {
    // Shared degradation profile
    const components = Array.from({ length: N }, () => ({
      shape: rand(1.5, 3.0),
      scale: rand(0.02, 0.06),
    }));

    // RCOA
    let rcoaHealth = Array(N).fill(1.0);
    const M_rcoa = 6;
    const crabTargets = Array(M_rcoa).fill(0);
    const crabPBest = Array(M_rcoa).fill(0);

    for (let t = 0; t < maxIters; t++) {
      // Degrade
      for (let i = 0; i < N; i++) {
        const deg = components[i].scale * Math.pow(-Math.log(1 - Math.random()), 1 / components[i].shape) * 0.6;
        rcoaHealth[i] = clamp(rcoaHealth[i] - deg, 0, 1);
      }

      // Density-penalized assignment
      const density = Array(N).fill(0);
      for (let j = 0; j < M_rcoa; j++) {
        let bestAttr = -Infinity, bestIdx = 0;
        for (let i = 0; i < N; i++) {
          const urgency = (1 - rcoaHealth[i]) * 2 + 0.5;
          const attr = urgency / (1 + 0.5 * density[i]);
          if (attr > bestAttr) { bestAttr = attr; bestIdx = i; }
        }
        crabTargets[j] = bestIdx;
        density[bestIdx]++;
      }

      // Fertilization (repair)
      for (let j = 0; j < M_rcoa; j++) {
        const idx = crabTargets[j];
        const densF = 1 / (1 + 0.5 * density[idx]);
        rcoaHealth[idx] = clamp(rcoaHealth[idx] + 0.25 * densF * (1 - rcoaHealth[idx]), 0, 1);
      }

      // Bioturbation: if stagnant, jolt
      for (let i = 0; i < N; i++) {
        if (rcoaHealth[i] < 0.3 && Math.random() < 0.1) {
          rcoaHealth[i] = clamp(rcoaHealth[i] + Math.abs(levyStep(1.5)) * 0.05, 0, 1);
        }
      }
    }
    const rcoaReliability = rcoaHealth.reduce((p, h) => p * Math.max(h, 0.01), 1);
    results.rcoa.push(rcoaHealth.reduce((s, h) => s + h, 0) / N);

    // PSO (no density penalty, fixed repair)
    let psoHealth = Array(N).fill(1.0);
    const psoTargets = Array(M_rcoa).fill(0);
    let psoGBest = 0;

    for (let t = 0; t < maxIters; t++) {
      for (let i = 0; i < N; i++) {
        const deg = components[i].scale * Math.pow(-Math.log(1 - Math.random()), 1 / components[i].shape) * 0.6;
        psoHealth[i] = clamp(psoHealth[i] - deg, 0, 1);
      }

      for (let j = 0; j < M_rcoa; j++) {
        const r1 = Math.random(), r2 = Math.random();
        let newTarget = Math.round(0.5 * psoTargets[j] + 0.3 * r1 * (crabPBest[j] - psoTargets[j]) + 0.3 * r2 * (psoGBest - psoTargets[j]));
        psoTargets[j] = clamp(Math.round(newTarget), 0, N - 1);
        psoHealth[psoTargets[j]] = clamp(psoHealth[psoTargets[j]] + 0.15, 0, 1);

        if (psoHealth[psoTargets[j]] > psoHealth[psoGBest]) psoGBest = psoTargets[j];
      }
    }
    results.pso.push(psoHealth.reduce((s, h) => s + h, 0) / N);

    // GA (binary encoding of repair assignments)
    let gaHealth = Array(N).fill(1.0);
    const gaPop = Array.from({ length: 20 }, () =>
      Array.from({ length: N }, () => Math.floor(Math.random() * 4)) // 0-3 repair levels
    );

    for (let t = 0; t < maxIters; t++) {
      for (let i = 0; i < N; i++) {
        const deg = components[i].scale * Math.pow(-Math.log(1 - Math.random()), 1 / components[i].shape) * 0.6;
        gaHealth[i] = clamp(gaHealth[i] - deg, 0, 1);
      }

      // Evaluate and apply best plan
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

      // Apply best plan (limited by M crews = 6 simultaneous repairs)
      let crewsUsed = 0;
      for (let i = 0; i < N && crewsUsed < M_rcoa; i++) {
        if (bestPlan[i] > 0) {
          gaHealth[i] = clamp(gaHealth[i] + [0, 0.05, 0.15, 0.3][bestPlan[i]], 0, 1);
          crewsUsed++;
        }
      }

      // Evolve
      for (let p = 0; p < gaPop.length; p++) {
        if (Math.random() < 0.3) {
          const donor = gaPop[Math.floor(Math.random() * gaPop.length)];
          gaPop[p] = gaPop[p].map((v, i) => Math.random() < 0.5 ? v : donor[i]);
        }
        gaPop[p] = gaPop[p].map(v => Math.random() < 0.05 ? Math.floor(Math.random() * 4) : v);
      }
    }
    results.ga.push(gaHealth.reduce((s, h) => s + h, 0) / N);
  }

  console.log("\nAvg Component Health after 500 iterations:");
  console.log(`  RCOA:  ${(mean(results.rcoa) * 100).toFixed(2)}% +/- ${(std(results.rcoa) * 100).toFixed(2)}%`);
  console.log(`  PSO:   ${(mean(results.pso) * 100).toFixed(2)}% +/- ${(std(results.pso) * 100).toFixed(2)}%`);
  console.log(`  GA:    ${(mean(results.ga) * 100).toFixed(2)}% +/- ${(std(results.ga) * 100).toFixed(2)}%`);

  const rcoaWinsPSO = results.rcoa.filter((v, i) => v > results.pso[i]).length;
  const rcoaWinsGA = results.rcoa.filter((v, i) => v > results.ga[i]).length;
  console.log(`  RCOA beats PSO: ${rcoaWinsPSO}/${runs} runs`);
  console.log(`  RCOA beats GA:  ${rcoaWinsGA}/${runs} runs`);

  return results;
}

// === Feature Selection Benchmark ===
function runFeatureSelectionBenchmark(runs = 21) {
  console.log("\n" + "=".repeat(70));
  console.log("  NOISY FEATURE SELECTION (12D total, 5 noise dims)");
  console.log("=".repeat(70));

  const D_total = 12, D_noise = 5, D_signal = D_total - D_noise;
  const nSignal = 80, nNoise = 120, nClusters = 3;
  const results = { rcoa: [], ga: [], pso: [] };

  for (let run = 0; run < runs; run++) {
    // Generate dataset
    const centers = Array.from({ length: nClusters }, () =>
      Array.from({ length: D_total }, (_, d) => d < D_signal ? rand(-2, 2) : 0)
    );

    const points = [];
    // Signal points
    for (let c = 0; c < nClusters; c++) {
      const perC = Math.floor(nSignal / nClusters);
      for (let i = 0; i < perC; i++) {
        const feats = centers[c].map((v, d) => d < D_signal ? v + gaussRand() * 0.5 : rand(-3, 3));
        points.push({ features: feats, isSignal: true, cluster: c });
      }
    }
    // Noise points
    for (let i = 0; i < nNoise; i++) {
      points.push({
        features: Array.from({ length: D_total }, () => rand(-4, 4)),
        isSignal: false, cluster: -1,
      });
    }

    // Fitness: classify using mask + centroids, penalize active noise dims
    function fitnessFS(mask, centroid) {
      let correct = 0;
      for (const p of points) {
        let d = 0;
        for (let k = 0; k < D_total; k++) {
          d += mask[k] * (p.features[k] - centroid[k]) ** 2;
        }
        d = Math.sqrt(d);
        const classified = d < 2.0;
        if ((p.isSignal && classified) || (!p.isSignal && !classified)) correct++;
      }
      let penalty = 0;
      for (let k = D_signal; k < D_total; k++) penalty += mask[k] * 0.05;
      return correct / points.length - penalty;
    }

    // RCOA with weeding
    let bestAcc = 0;
    const rcoaCentroids = Array.from({ length: 8 }, () => ({
      features: Array.from({ length: D_total }, () => rand(-2, 2)),
      mask: Array(D_total).fill(1),
    }));

    for (let t = 0; t < 200; t++) {
      for (const rc of rcoaCentroids) {
        const fit = fitnessFS(rc.mask, rc.features);

        // Weeding: test random D/3 dimensions
        for (let k = 0; k < 4; k++) {
          const d = Math.floor(Math.random() * D_total);
          if (rc.mask[d] === 0) continue;
          const trialMask = rc.mask.slice();
          trialMask[d] = 0;
          const trialFit = fitnessFS(trialMask, rc.features);
          if (trialFit > fit) {
            rc.mask[d] = 0;
          }
        }

        // Fertilization: nudge toward better
        const curFit = fitnessFS(rc.mask, rc.features);
        for (let d = 0; d < D_total; d++) {
          if (rc.mask[d] === 0) continue;
          rc.features[d] += 0.1 * gaussRand() * 0.3;
        }

        const newFit = fitnessFS(rc.mask, rc.features);
        if (newFit > bestAcc) bestAcc = newFit;
      }
    }
    const rcoaWeeded = rcoaCentroids[0].mask.filter((v, i) => v === 0 && i >= D_signal).length;
    results.rcoa.push({ acc: bestAcc, noiseDimsWeeded: rcoaWeeded });

    // GA (binary mask, no sensitivity-guided weeding)
    let gaBestAcc = 0;
    let gaPop2 = Array.from({ length: 16 }, () => ({
      mask: Array.from({ length: D_total }, () => Math.random() > 0.3 ? 1 : 0),
      features: Array.from({ length: D_total }, () => rand(-2, 2)),
    }));

    for (let t = 0; t < 200; t++) {
      const fits = gaPop2.map(ind => fitnessFS(ind.mask, ind.features));
      const maxFit = Math.max(...fits);
      if (maxFit > gaBestAcc) gaBestAcc = maxFit;

      const newPop = [];
      for (let i = 0; i < gaPop2.length; i++) {
        const a = Math.floor(Math.random() * gaPop2.length);
        const b = Math.floor(Math.random() * gaPop2.length);
        const parent = fits[a] > fits[b] ? gaPop2[a] : gaPop2[b];
        newPop.push({
          mask: parent.mask.map(v => Math.random() < 0.05 ? (1 - v) : v),
          features: parent.features.map(f => f + gaussRand() * 0.1),
        });
      }
      gaPop2 = newPop;
    }
    results.ga.push({ acc: gaBestAcc, noiseDimsWeeded: 0 });

    // PSO (continuous weights)
    let psoBestAcc = 0;
    let psoParticles = Array.from({ length: 16 }, () => ({
      mask: Array.from({ length: D_total }, () => rand(0, 1)),
      features: Array.from({ length: D_total }, () => rand(-2, 2)),
      v: Array.from({ length: D_total }, () => rand(-0.1, 0.1)),
    }));

    for (let t = 0; t < 200; t++) {
      for (const p of psoParticles) {
        const roundedMask = p.mask.map(v => v > 0.5 ? 1 : 0);
        const fit = fitnessFS(roundedMask, p.features);
        if (fit > psoBestAcc) psoBestAcc = fit;
        // Simple PSO update
        p.v = p.v.map((v, d) => 0.7 * v + 0.3 * Math.random() * (rand(-0.5, 0.5)));
        p.mask = p.mask.map((m, d) => clamp(m + p.v[d], 0, 1));
        p.features = p.features.map(f => f + gaussRand() * 0.05);
      }
    }
    results.pso.push({ acc: psoBestAcc, noiseDimsWeeded: 0 });
  }

  console.log("\nClassification Accuracy:");
  console.log(`  RCOA:  ${(mean(results.rcoa.map(r => r.acc)) * 100).toFixed(2)}% (avg ${mean(results.rcoa.map(r => r.noiseDimsWeeded)).toFixed(1)}/${D_noise} noise dims weeded)`);
  console.log(`  GA:    ${(mean(results.ga.map(r => r.acc)) * 100).toFixed(2)}%`);
  console.log(`  PSO:   ${(mean(results.pso.map(r => r.acc)) * 100).toFixed(2)}%`);

  return results;
}

// === Main ===
function main() {
  console.log("=".repeat(70));
  console.log("  RCOA HEADLESS BENCHMARK SUITE");
  console.log("  Rice-Crab Optimization Algorithm vs PSO, DE, GA");
  console.log("=".repeat(70));

  const dims = [10, 30];
  const runs = 11;
  const allResults = {};

  for (const [key, bench] of Object.entries(benchmarks)) {
    for (const D of dims) {
      const maxFEs = D * 5000;
      console.log(`\n--- ${bench.name} (D=${D}, maxFEs=${maxFEs}, ${runs} runs) ---`);

      const rcoaRuns = [], psoRuns = [], deRuns = [], gaRuns = [];

      for (let r = 0; r < runs; r++) {
        const rcoaParams = bench.hasNoisyDims ?
          { N: 20, M: 15, weedEnabled: true, epsilon: 0.1 } :
          { N: 20, M: 15, weedEnabled: false };

        const rcoa = runRCOA(bench.fn, D, [bench.bounds[0], bench.bounds[1]], maxFEs, rcoaParams);
        const pso = runPSO(bench.fn, D, [bench.bounds[0], bench.bounds[1]], maxFEs);
        const de = runDE(bench.fn, D, [bench.bounds[0], bench.bounds[1]], maxFEs);
        const ga = runGA(bench.fn, D, [bench.bounds[0], bench.bounds[1]], maxFEs);

        rcoaRuns.push(rcoa.bestFit);
        psoRuns.push(pso.bestFit);
        deRuns.push(de.bestFit);
        gaRuns.push(ga.bestFit);
      }

      const pad = (s, n) => s.padEnd(n);
      console.log(`  ${pad("RCOA:", 8)} mean=${mean(rcoaRuns).toExponential(3)} std=${std(rcoaRuns).toExponential(3)}`);
      console.log(`  ${pad("PSO:", 8)} mean=${mean(psoRuns).toExponential(3)} std=${std(psoRuns).toExponential(3)}`);
      console.log(`  ${pad("DE:", 8)} mean=${mean(deRuns).toExponential(3)} std=${std(deRuns).toExponential(3)}`);
      console.log(`  ${pad("GA:", 8)} mean=${mean(gaRuns).toExponential(3)} std=${std(gaRuns).toExponential(3)}`);

      // Simple win count
      const rcoaWins = {
        vsPSO: rcoaRuns.filter((v, i) => v < psoRuns[i]).length,
        vsDE: rcoaRuns.filter((v, i) => v < deRuns[i]).length,
        vsGA: rcoaRuns.filter((v, i) => v < gaRuns[i]).length,
      };
      console.log(`  RCOA wins: vs PSO ${rcoaWins.vsPSO}/${runs}, vs DE ${rcoaWins.vsDE}/${runs}, vs GA ${rcoaWins.vsGA}/${runs}`);

      allResults[`${key}_D${D}`] = { rcoaRuns, psoRuns, deRuns, gaRuns };
    }
  }

  // Application benchmarks
  runSMPBenchmark(runs);
  runFeatureSelectionBenchmark(runs);

  console.log("\n" + "=".repeat(70));
  console.log("  BENCHMARK COMPLETE");
  console.log("=".repeat(70));
}

main();
