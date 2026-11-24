import { Container, Graphics, Text, TextStyle } from 'pixi.js';
import type { DeskPlacement, OfficeLayoutModel } from './officeLayout';
import type { PixelAgentSprite } from './agentSprite';

export interface AnimatedEnvelope {
  graphic: Container;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
}

export function buildDimensionOverlayLayer(
  layout: OfficeLayoutModel,
  isBaseline: boolean
): { container: Container; envelopes: AnimatedEnvelope[] } {
  const container = new Container();
  const envelopes: AnimatedEnvelope[] = [];
  const deskById = new Map<string, DeskPlacement>(layout.desks.map((d) => [d.agent.agent_id, d]));

  // 1. Parallel cluster highlights (only shown when Workflow Parallelization is confirmed active)
  for (const cluster of layout.parallelClusters) {
    const g = new Graphics();
    g.roundRect(cluster.bounds.x, cluster.bounds.y, cluster.bounds.w, cluster.bounds.h, 8)
      .fill({ color: 0x2b7c7e, alpha: 0.08 })
      .stroke({ width: 2, color: 0x2b7c7e });
    container.addChild(g);

    const tag = new Text({
      text: `PARALLEL GROUP ${cluster.groupIndex} (${cluster.agentIds.length} AGENTS CONCURRENT)`,
      style: new TextStyle({
        fontFamily: 'IBM Plex Mono, monospace',
        fontSize: 9,
        fontWeight: '700',
        fill: 0x1d5b5d,
      }),
    });
    tag.x = cluster.bounds.x + 8;
    tag.y = cluster.bounds.y - 14;
    container.addChild(tag);
  }

  // 2. Data handoff flow lines & animated packets (full heavy packets on baseline, slim compact packets when pruned)
  const pathsG = new Graphics();
  for (const edge of layout.handoffEdges) {
    const src = deskById.get(edge.fromAgentId);
    const dst = deskById.get(edge.toAgentId);
    if (!src || !dst) continue;

    const pruned = !isBaseline && edge.removedFields.length > 0;
    pathsG.moveTo(src.x, src.y);
    pathsG.lineTo(dst.x, dst.y);
    pathsG.stroke({
      width: pruned ? 2 : 3,
      color: pruned ? 0x246a49 : 0xc99c5e,
      alpha: 0.55,
    });

    // Animated envelope sprite moving along the handoff edge
    const env = new Container();
    const eg = new Graphics();
    if (pruned) {
      // Compact trimmed emerald packet
      eg.rect(-8, -5, 16, 10).fill(0x4cb87f).stroke({ width: 1.5, color: 0x1f3a2e });
    } else {
      // Thick bulky baseline amber payload packet
      eg.rect(-11, -7, 22, 14).fill(0xd9822b).stroke({ width: 1.5, color: 0x1f3a2e });
      eg.rect(-9, -10, 18, 3).fill(0xfffdf6).stroke({ width: 1, color: 0x1f3a2e });
    }
    env.addChild(eg);

    if (pruned) {
      const badge = new Text({
        text: `-${edge.removedFields.length}f`,
        style: new TextStyle({
          fontFamily: 'IBM Plex Mono, monospace',
          fontSize: 8,
          fontWeight: '700',
          fill: 0x1b5c3b,
        }),
      });
      badge.x = 10;
      badge.y = -6;
      env.addChild(badge);
    }

    container.addChild(env);
    envelopes.push({
      graphic: env,
      fromX: src.x,
      fromY: src.y,
      toX: dst.x,
      toY: dst.y,
      progress: (envelopes.length * 0.23) % 1,
      speed: isBaseline ? 0.0022 : pruned ? 0.0046 : 0.0032,
    });
  }
  container.addChildAt(pathsG, 0);

  // 3. Prompt optimization tags & consolidation arrows
  for (const desk of layout.desks) {
    if (desk.promptCompressedPct !== null && !desk.consolidatedInto) {
      const pill = new Graphics();
      pill.rect(desk.x - 42, desk.y - 44, 84, 12).fill(0xdff2e3).stroke({ width: 1, color: 0x1b5c3b });
      const txt = new Text({
        text: `PROMPT -${Math.round(desk.promptCompressedPct)}%`,
        style: new TextStyle({
          fontFamily: 'IBM Plex Mono, monospace',
          fontSize: 7,
          fontWeight: '700',
          fill: 0x1b5c3b,
        }),
      });
      txt.x = desk.x - 36;
      txt.y = desk.y - 42;
      container.addChild(pill, txt);
    }
  }

  return { container, envelopes };
}

export function stepOfficeAnimations(
  sprites: PixelAgentSprite[],
  envelopes: AnimatedEnvelope[],
  elapsedMs: number,
  isBaseline: boolean
): void {
  const t = elapsedMs * 0.005;

  for (const sp of sprites) {
    const phase = t + sp.phaseOffset;
    if (isBaseline) {
      // Static historical office snapshot — very faint breathing only
      sp.bodyGroup.y = Math.round(Math.sin(phase * 0.4) * 0.6);
      sp.halo.alpha = 0.35;
      continue;
    }

    switch (sp.state) {
      case 'working':
      case 'thinking':
        sp.bodyGroup.y = Math.round(Math.sin(phase * 2.2) * 1.6);
        sp.halo.alpha = 0.55 + 0.35 * Math.sin(phase * 2.5);
        break;
      case 'parallel_working':
        // Synchronized concurrent pulse across parallel cohort
        sp.bodyGroup.y = Math.round(Math.sin(t * 2.8) * 2.2);
        sp.halo.alpha = 0.65 + 0.35 * Math.sin(t * 3.0);
        break;
      case 'moving':
        sp.bodyGroup.x = Math.round(Math.sin(phase * 2.0) * 6);
        sp.bodyGroup.y = Math.round(Math.abs(Math.sin(phase * 3.0)) * -2);
        break;
      case 'failed':
      case 'blocked':
        sp.bodyGroup.x = Math.round(Math.sin(phase * 6.0) * 1.2);
        sp.halo.alpha = 0.7 + 0.3 * Math.sin(phase * 4.0);
        break;
      default:
        sp.bodyGroup.x = 0;
        sp.bodyGroup.y = Math.round(Math.sin(phase * 0.8) * 1);
        break;
    }
  }

  for (const env of envelopes) {
    env.progress = (env.progress + env.speed) % 1;
    env.graphic.x = env.fromX + (env.toX - env.fromX) * env.progress;
    env.graphic.y = env.fromY + (env.toY - env.fromY) * env.progress - Math.sin(env.progress * Math.PI) * 12;
  }
}
