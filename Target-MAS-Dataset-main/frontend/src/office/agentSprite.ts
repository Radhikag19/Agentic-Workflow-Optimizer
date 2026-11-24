import { Container, Graphics, Text, TextStyle } from 'pixi.js';
import type { AgentVisualState } from '../types/workplace';

const SHIRT_PALETTE = [
  0x3b5e8c, // slate blue
  0x246a49, // forest green
  0xd9822b, // warm amber
  0x6d4c8f, // plum
  0x2b7c7e, // teal
  0xb85c38, // terracotta
  0x4c6f9c, // denim
  0x5e7c3e, // moss
];

const HAIR_PALETTE = [0x3b2a1e, 0x5c3c24, 0x222222, 0x8a5a32, 0x4e3529];

function hashString(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h);
}

const STATE_BADGE_COLORS: Record<AgentVisualState, number> = {
  idle: 0x9aa39a,
  thinking: 0x6d4c8f,
  working: 0x246a49,
  waiting: 0xd9822b,
  moving: 0x3b5e8c,
  parallel_working: 0x2b7c7e,
  completed: 0x2b7a54,
  failed: 0xb83b35,
  blocked: 0x8f221d,
};

const STATE_LABELS: Record<AgentVisualState, string> = {
  idle: 'IDLE',
  thinking: 'THINK',
  working: 'LIVE',
  waiting: 'WAIT',
  moving: 'MOVE',
  parallel_working: 'SYNC//',
  completed: 'DONE',
  failed: 'FAIL!',
  blocked: 'STOP!',
};

export interface PixelAgentSprite {
  container: Container;
  bodyGroup: Container;
  halo: Graphics;
  bubbleGroup: Container;
  agentId: string;
  baseX: number;
  baseY: number;
  state: AgentVisualState;
  phaseOffset: number;
}

export function createPixelAgentSprite(params: {
  agentId: string;
  x: number;
  y: number;
  state: AgentVisualState;
  modelTier: 'heavy' | 'medium' | 'light';
  speechText?: string;
  isBaseline: boolean;
  onSelect: (agentId: string) => void;
}): PixelAgentSprite {
  const { agentId, x, y, state, modelTier, speechText, isBaseline, onSelect } = params;
  const h = hashString(agentId);
  const shirtColor = SHIRT_PALETTE[h % SHIRT_PALETTE.length];
  const hairColor = HAIR_PALETTE[(h >> 3) % HAIR_PALETTE.length];

  const container = new Container();
  container.x = x;
  container.y = y;
  container.eventMode = 'static';
  container.cursor = 'pointer';
  container.on('pointertap', () => onSelect(agentId));

  // Status halo behind sprite
  const halo = new Graphics();
  const haloColor = STATE_BADGE_COLORS[state] ?? 0x9aa39a;
  if (state === 'working' || state === 'parallel_working' || state === 'failed' || state === 'blocked') {
    halo.circle(0, 4, 22).fill({ color: haloColor, alpha: 0.28 }).stroke({ width: 2, color: haloColor });
  }
  container.addChild(halo);

  // Body sprite (pixel character built with clean rectangles)
  const bodyGroup = new Container();
  const g = new Graphics();

  // Equipment pack prop reflecting model tier (NO body-weight stereotypes!)
  if (modelTier === 'heavy') {
    // Twin compute server pack (120B / 70B)
    g.rect(-18, -2, 8, 18).fill(0xd9822b).stroke({ width: 1, color: 0x1f3a2e });
    g.rect(10, -2, 8, 18).fill(0xd9822b).stroke({ width: 1, color: 0x1f3a2e });
  } else if (modelTier === 'light') {
    // Sleek compact lightning satchel (8B)
    g.rect(10, 2, 7, 11).fill(0x4cb87f).stroke({ width: 1, color: 0x1f3a2e });
  }

  // Legs
  g.rect(-7, 14, 5, 7).fill(0x2d333b);
  g.rect(2, 14, 5, 7).fill(0x2d333b);

  // Torso (shirt)
  g.rect(-10, 0, 20, 15).fill(shirtColor).stroke({ width: 2, color: 0x1f3a2e });

  // Arms
  g.rect(-13, 1, 4, 10).fill(shirtColor).stroke({ width: 1, color: 0x1f3a2e });
  g.rect(9, 1, 4, 10).fill(shirtColor).stroke({ width: 1, color: 0x1f3a2e });

  // Head
  g.rect(-8, -14, 16, 14).fill(0xf3cbb0).stroke({ width: 2, color: 0x1f3a2e });
  // Hair cap
  g.rect(-9, -17, 18, 6).fill(hairColor).stroke({ width: 1, color: 0x1f3a2e });

  // Eyes (failed -> crimson X style, otherwise dark eyes)
  if (state === 'failed' || state === 'blocked') {
    g.rect(-5, -9, 3, 3).fill(0xb83b35);
    g.rect(2, -9, 3, 3).fill(0xb83b35);
  } else {
    g.rect(-5, -9, 2, 3).fill(0x1f3a2e);
    g.rect(3, -9, 2, 3).fill(0x1f3a2e);
  }

  bodyGroup.addChild(g);
  container.addChild(bodyGroup);

  // Compact overhead state badge + speech bubble
  const bubbleGroup = new Container();
  bubbleGroup.y = -32;

  const badgeColor = isBaseline ? 0x665b4c : STATE_BADGE_COLORS[state] ?? 0x9aa39a;
  const labelText = isBaseline
    ? 'HIST'
    : speechText
    ? `${STATE_LABELS[state]} ${speechText.slice(0, 14)}`
    : STATE_LABELS[state];

  const badgeW = Math.min(118, Math.max(40, labelText.length * 6 + 10));
  const bg = new Graphics();
  bg.rect(-badgeW / 2, -12, badgeW, 14).fill(0xfffdf6).stroke({ width: 2, color: badgeColor });
  bg.rect(-badgeW / 2, -12, 6, 14).fill(badgeColor);
  bubbleGroup.addChild(bg);

  const txt = new Text({
    text: labelText,
    style: new TextStyle({
      fontFamily: 'IBM Plex Mono, Consolas, monospace',
      fontSize: 8,
      fontWeight: '700',
      fill: 0x1f3a2e,
    }),
  });
  txt.x = -badgeW / 2 + 8;
  txt.y = -10;
  bubbleGroup.addChild(txt);

  container.addChild(bubbleGroup);

  return {
    container,
    bodyGroup,
    halo,
    bubbleGroup,
    agentId,
    baseX: x,
    baseY: y,
    state,
    phaseOffset: (h % 100) / 15,
  };
}
