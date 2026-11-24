import { Application, Container, Text, TextStyle } from 'pixi.js';
import { createPixelAgentSprite, type PixelAgentSprite } from './agentSprite';
import type { OfficeLayoutModel } from './officeLayout';
import { buildOfficeFloorLayer, createDeskContainer } from './officeStations';
import {
  buildDimensionOverlayLayer,
  stepOfficeAnimations,
  type AnimatedEnvelope,
} from './officeAnimations';

export class OfficeSceneController {
  private app: Application | null = null;
  private rootStage = new Container();
  private sprites: PixelAgentSprite[] = [];
  private envelopes: AnimatedEnvelope[] = [];
  private elapsedMs = 0;
  private isBaseline = false;
  private disposed = false;
  private initialized = false;
  private pendingLayout: {
    layout: OfficeLayoutModel;
    selectedAgentId: string | null;
    isBaseline: boolean;
    onSelectAgent: (id: string) => void;
  } | null = null;

  public async mount(hostElement: HTMLElement, width: number, height: number): Promise<void> {
    const app = new Application();
    this.app = app;
    await app.init({
      width,
      height,
      backgroundColor: 0xf5efe0,
      antialias: false,
      resolution: 1,
      preference: 'webgl',
    });

    if (this.disposed) {
      app.destroy(true, { children: true });
      this.app = null;
      return;
    }

    hostElement.innerHTML = '';
    app.canvas.style.width = '100%';
    app.canvas.style.height = 'auto';
    app.canvas.style.imageRendering = 'pixelated';
    hostElement.appendChild(app.canvas);

    app.stage.addChild(this.rootStage);
    app.ticker.add((ticker) => {
      this.elapsedMs += ticker.deltaMS;
      stepOfficeAnimations(this.sprites, this.envelopes, this.elapsedMs, this.isBaseline);
    });
    this.initialized = true;

    if (this.pendingLayout) {
      const pending = this.pendingLayout;
      this.pendingLayout = null;
      this.renderLayout(
        pending.layout,
        pending.selectedAgentId,
        pending.isBaseline,
        pending.onSelectAgent
      );
    }
  }

  public renderLayout(
    layout: OfficeLayoutModel,
    selectedAgentId: string | null,
    isBaseline: boolean,
    onSelectAgent: (id: string) => void
  ): void {
    if (!this.initialized || !this.app) {
      this.pendingLayout = { layout, selectedAgentId, isBaseline, onSelectAgent };
      return;
    }

    this.isBaseline = isBaseline;
    this.sprites = [];
    this.envelopes = [];
    this.rootStage.removeChildren();

    // 1. Floor, walls, decorative plants & shared office stations
    this.rootStage.addChild(buildOfficeFloorLayer(layout, isBaseline));

    // 2. Dimension overlays (parallel groups, handoff paths, envelopes, prompt badges)
    const overlays = buildDimensionOverlayLayer(layout, isBaseline);
    this.envelopes = overlays.envelopes;
    this.rootStage.addChild(overlays.container);

    // 3. Desks & clickable characters
    for (const desk of layout.desks) {
      const selected = desk.agent.agent_id === selectedAgentId;
      this.rootStage.addChild(createDeskContainer(desk, selected, onSelectAgent));

      if (!desk.consolidatedInto) {
        const sprite = createPixelAgentSprite({
          agentId: desk.agent.agent_id,
          x: desk.chairX,
          y: desk.chairY,
          state: desk.telemetry.state,
          modelTier: desk.modelTier,
          speechText: desk.telemetry.speech_bubble,
          isBaseline,
          onSelect: onSelectAgent,
        });
        this.sprites.push(sprite);
        this.rootStage.addChild(sprite.container);
      }
    }

    // 4. Footer status strip inside canvas
    const footer = new Text({
      text: `${layout.modeLabel} • ${layout.desks.length} AGENTS LOADED`,
      style: new TextStyle({
        fontFamily: 'IBM Plex Mono, Consolas, monospace',
        fontSize: 10,
        fontWeight: '700',
        fill: 0x1f3a2e,
      }),
    });
    footer.x = 20;
    footer.y = layout.height - 26;
    this.rootStage.addChild(footer);
  }

  public destroy(): void {
    this.disposed = true;
    this.initialized = false;
    this.sprites = [];
    this.envelopes = [];
    if (this.app) {
      try {
        this.app.destroy(true, { children: true });
      } catch {
        // ignore double destroy
      }
      this.app = null;
    }
  }
}
