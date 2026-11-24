import { Container, Graphics, Text, TextStyle } from 'pixi.js';
import type { DeskPlacement, OfficeLayoutModel, StationRegion } from './officeLayout';

const LABEL_STYLE = new TextStyle({
  fontFamily: 'IBM Plex Mono, Consolas, monospace',
  fontSize: 9,
  fontWeight: '700',
  fill: 0x1f3a2e,
});

const SUB_STYLE = new TextStyle({
  fontFamily: 'IBM Plex Mono, Consolas, monospace',
  fontSize: 8,
  fontWeight: '600',
  fill: 0x3b2a1e,
});

export function buildOfficeFloorLayer(layout: OfficeLayoutModel, isBaseline: boolean): Container {
  const layer = new Container();

  // Checkered retro office carpet floor
  const floor = new Graphics();
  const tile = 24;
  for (let y = 0; y < layout.height; y += tile) {
    for (let x = 0; x < layout.width; x += tile) {
      const even = ((x / tile) + (y / tile)) % 2 === 0;
      const color = isBaseline
        ? even
          ? 0xdfd8c5
          : 0xd5ceba
        : even
        ? 0xdce4d6
        : 0xd1dac9;
      floor.rect(x, y, tile, tile).fill(color);
    }
  }

  // Timber perimeter wall + wainscoting
  floor.rect(0, 0, layout.width, 16).fill(0x7e5630);
  floor.rect(0, 16, layout.width, 6).fill(0xb98b5a);
  floor.rect(0, layout.height - 10, layout.width, 10).fill(0x7e5630);
  floor.rect(0, 0, 10, layout.height).fill(0x7e5630);
  floor.rect(layout.width - 10, 0, 10, layout.height).fill(0x7e5630);
  layer.addChild(floor);

  // Decorative office plants at corners
  const plants: [number, number][] = [
    [22, 558],
    [668, 165],
    [916, 178],
  ];
  for (const [px, py] of plants) {
    layer.addChild(createPixelPlant(px, py));
  }

  // Station regions (Entrance, Task Board, Model Rack, Document Station, Manager Desk, Meeting Room)
  for (const st of layout.stations) {
    layer.addChild(createStationNode(st, layout));
  }

  return layer;
}

function createPixelPlant(x: number, y: number): Container {
  const c = new Container();
  c.x = x;
  c.y = y;
  const g = new Graphics();
  // Terracotta pot
  g.rect(-8, 4, 16, 12).fill(0xb85c38).stroke({ width: 2, color: 0x1f3a2e });
  // Pixel leaves
  g.rect(-12, -8, 10, 10).fill(0x2d7a4f);
  g.rect(2, -10, 10, 10).fill(0x3a9460);
  g.rect(-6, -16, 12, 12).fill(0x246a49);
  c.addChild(g);
  return c;
}

function createStationNode(st: StationRegion, layout: OfficeLayoutModel): Container {
  const c = new Container();
  c.x = st.x;
  c.y = st.y;

  const g = new Graphics();
  const bg =
    st.id === 'meeting_room'
      ? 0xe8dec5
      : st.id === 'manager_office'
      ? 0xe7d6ba
      : 0xf4ecd8;
  g.rect(0, 0, st.w, st.h).fill(bg).stroke({ width: 2, color: 0x1f3a2e });
  g.rect(0, 0, st.w, 18).fill(0xcab68d).stroke({ width: 2, color: 0x1f3a2e });

  if (st.id === 'entrance') {
    g.rect(18, 36, 46, 26).fill(0x9a6d43).stroke({ width: 2, color: 0x1f3a2e });
    g.rect(82, 34, 44, 34).fill(0xfffdf6).stroke({ width: 2, color: 0x1f3a2e });
    // Paper inbox stack
    g.rect(90, 42, 28, 6).fill(0xd9822b);
    g.rect(90, 52, 28, 6).fill(0x3b5e8c);
  } else if (st.id === 'task_board') {
    // Corkboard with pinned sticky notes
    g.rect(14, 28, st.w - 28, 56).fill(0xd4b886).stroke({ width: 2, color: 0x7e5630 });
    const noteColors = [0xf7e379, 0x99d6b3, 0xf2b591, 0xa8c8e8];
    for (let i = 0; i < 4; i++) {
      g.rect(24 + i * 34, 38, 24, 22).fill(noteColors[i]).stroke({ width: 1, color: 0x1f3a2e });
    }
  } else if (st.id === 'model_rack') {
    // Hardware rack visual comparing heavy vs nimble compute
    g.rect(18, 28, 52, 58).fill(0x2d3b42).stroke({ width: 2, color: 0x1f3a2e });
    g.rect(24, 34, 40, 8).fill(0xd9822b);
    g.rect(24, 46, 40, 8).fill(0xd9822b);
    g.rect(24, 58, 40, 8).fill(0x4cb87f);
    const tierCounts = layout.desks.reduce(
      (acc, d) => {
        acc[d.modelTier] += 1;
        return acc;
      },
      { heavy: 0, medium: 0, light: 0 }
    );
    const rackTxt = new Text({
      text: `120B:${tierCounts.heavy}\n8B:${tierCounts.light}`,
      style: SUB_STYLE,
    });
    rackTxt.x = 80;
    rackTxt.y = 36;
    c.addChild(rackTxt);
  } else if (st.id === 'doc_station') {
    // Filing cabinets + shredder bin for data handoff pruning
    g.rect(16, 30, 42, 54).fill(0x9fb0b5).stroke({ width: 2, color: 0x1f3a2e });
    g.rect(22, 36, 30, 14).fill(0xc4d0d4).stroke({ width: 1, color: 0x1f3a2e });
    g.rect(22, 56, 30, 14).fill(0xc4d0d4).stroke({ width: 1, color: 0x1f3a2e });
    // Pruner / compact envelope station
    g.rect(74, 34, 72, 48).fill(0xfdf8ec).stroke({ width: 2, color: 0x1f3a2e });
    const prunedCount = layout.handoffEdges.reduce((n, e) => n + e.removedFields.length, 0);
    const docTxt = new Text({
      text: prunedCount > 0 ? `PRUNED:\n${prunedCount} fields` : 'FULL\nPAYLOADS',
      style: SUB_STYLE,
    });
    docTxt.x = 82;
    docTxt.y = 44;
    c.addChild(docTxt);
  } else if (st.id === 'meeting_room') {
    // Conference table & chairs
    g.roundRect(44, 54, 154, 78, 12).fill(0xb98b5a).stroke({ width: 2, color: 0x1f3a2e });
    for (let i = 0; i < 4; i++) {
      g.rect(58 + i * 36, 38, 22, 12).fill(0x3b5e8c).stroke({ width: 1, color: 0x1f3a2e });
      g.rect(58 + i * 36, 136, 22, 12).fill(0x3b5e8c).stroke({ width: 1, color: 0x1f3a2e });
    }
    // Whiteboard on wall
    g.rect(18, 158, 208, 18).fill(0xfffdf6).stroke({ width: 2, color: 0x1f3a2e });
    const boardTxt = new Text({
      text:
        layout.parallelClusters.length > 0
          ? `PARALLEL STAGES ACTIVE: ${layout.parallelClusters.length}`
          : 'SEQUENTIAL WORKFLOW TRACE',
      style: SUB_STYLE,
    });
    boardTxt.x = 26;
    boardTxt.y = 161;
    c.addChild(boardTxt);
  }

  c.addChildAt(g, 0);

  const label = new Text({ text: st.label, style: LABEL_STYLE });
  label.x = 8;
  label.y = 4;
  c.addChild(label);

  return c;
}

export function createDeskContainer(
  desk: DeskPlacement,
  selected: boolean,
  onSelect: (agentId: string) => void
): Container {
  const container = new Container();
  container.x = desk.x;
  container.y = desk.y;
  container.eventMode = 'static';
  container.cursor = 'pointer';
  container.on('pointertap', () => onSelect(desk.agent.agent_id));

  const g = new Graphics();

  // If consolidated away cleanly, show neat empty "Merged" desk plaque
  if (desk.consolidatedInto) {
    g.rect(-46, -24, 92, 46).fill(0xe4dfd0).stroke({ width: 2, color: 0x8c8476 });
    const mergedText = new Text({
      text: `MERGED ->\n${desk.consolidatedInto.slice(0, 12)}`,
      style: new TextStyle({
        fontFamily: 'IBM Plex Mono, monospace',
        fontSize: 8,
        fontWeight: '700',
        fill: 0x665b4c,
      }),
    });
    mergedText.x = -38;
    mergedText.y = -12;
    container.addChild(g, mergedText);
    return container;
  }

  // Highlight border if selected
  if (selected) {
    g.rect(-56, -34, 112, 74).stroke({ width: 3, color: 0xd9822b });
  }

  // Desk wood surface
  const deskColor = desk.isManager ? 0x9b6637 : 0xc59b6d;
  g.rect(-48, -18, 96, 36).fill(deskColor).stroke({ width: 2, color: 0x1f3a2e });

  // Retro CRT Monitor
  const screenActive = desk.telemetry.state === 'working' || desk.telemetry.state === 'parallel_working';
  const screenFailed = desk.telemetry.state === 'failed' || desk.telemetry.state === 'blocked';
  const screenColor = screenFailed ? 0xe05850 : screenActive ? 0x6ef0a7 : 0x45695b;
  g.rect(-18, -26, 36, 20).fill(0xd9d4c5).stroke({ width: 2, color: 0x1f3a2e });
  g.rect(-14, -23, 28, 13).fill(screenColor);

  // Paper stack prop: thinner when prompt optimized or handoff pruned, taller on baseline
  const stackHeight =
    desk.promptCompressedPct !== null || desk.removedFields.length > 0 ? 5 : 11;
  g.rect(-42, -12, 16, stackHeight).fill(0xfffdf6).stroke({ width: 1, color: 0x1f3a2e });

  // Prompt instruction card prop (compact emerald card when prompt optimized, thick binder otherwise)
  const manualColor = desk.promptCompressedPct !== null ? 0x3ba16d : 0xd9822b;
  g.rect(24, -12, 16, desk.promptCompressedPct !== null ? 6 : 12)
    .fill(manualColor)
    .stroke({ width: 1, color: 0x1f3a2e });

  // Office Chair behind desk
  g.rect(-12, 20, 24, 14).fill(0x3b5e8c).stroke({ width: 2, color: 0x1f3a2e });

  // Multi-agent absorbed badge
  if (desk.absorbedAgents.length > 0) {
    g.rect(28, -32, 26, 12).fill(0x6d4c8f).stroke({ width: 1, color: 0x1f3a2e });
    const absTxt = new Text({
      text: `+${desk.absorbedAgents.length}`,
      style: new TextStyle({ fontFamily: 'IBM Plex Mono', fontSize: 8, fill: 0xffffff, fontWeight: '700' }),
    });
    absTxt.x = 33;
    absTxt.y = -31;
    container.addChild(absTxt);
  }

  container.addChildAt(g, 0);

  // Desk nameplate
  const shortName =
    desk.agent.agent_name.length > 16
      ? `${desk.agent.agent_name.slice(0, 15)}…`
      : desk.agent.agent_name;
  const plate = new Text({
    text: shortName,
    style: new TextStyle({
      fontFamily: 'IBM Plex Mono, Consolas, monospace',
      fontSize: 9,
      fontWeight: '700',
      fill: 0x1f3a2e,
    }),
  });
  plate.anchor.set(0.5, 0);
  plate.x = 0;
  plate.y = 37;
  container.addChild(plate);

  return container;
}
