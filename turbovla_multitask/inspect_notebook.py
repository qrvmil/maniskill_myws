"""Render exported notebook offline and capture its actual figures/table/video."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
PREVIEW=ROOT/'docs/turbovla_multitask/preview'


def inspect():
    source=PREVIEW/'02_turbovla_multitask_eval.html'
    if not source.exists(): raise FileNotFoundError('Execute notebook and export HTML first')
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
        page.route('http://**/*',lambda route:route.abort())
        page.route('https://**/*',lambda route:route.abort())
        page.goto(source.as_uri(),wait_until='domcontentloaded')
        images=page.locator('img[src^="data:image/png"]')
        for index in range(images.count()):
            images.nth(index).wait_for(state='visible')
            images.nth(index).screenshot(path=str(PREVIEW/f'figure_{index}.png'))
        tables=page.locator('table.dataframe')
        if tables.count()<3: raise ValueError('Model selection, episode and complete matrix tables must render')
        tables.last.screenshot(path=str(PREVIEW/'matrix_table.png'))
        page.evaluate('window.scrollTo(0, 0)')
        page.screenshot(path=str(PREVIEW/'notebook_top.png'))
        videos=page.locator('video')
        if videos.count()!=1: raise ValueError('Exactly one embedded demonstration video required')
        video=videos.first
        page.wait_for_function('document.querySelector("video").readyState >= 2',timeout=20000)
        video.evaluate('(v) => {v.currentTime = Math.min(2, v.duration / 2)}')
        page.wait_for_timeout(800)
        media=video.evaluate('(v) => ({width:v.videoWidth,height:v.videoHeight,duration:v.duration,readyState:v.readyState,currentTime:v.currentTime})')
        if not media['width'] or media['duration']<=0: raise ValueError('Embedded video did not decode')
        video.screenshot(path=str(PREVIEW/'video_frame.png'))
        status=dict(browser='Chromium headless',offline=True,viewport=[1280,900],
            headings=page.locator('h1,h2,h3').all_text_contents(),png_figures=images.count(),
            dataframe_tables=tables.count(),table_rows=[t.locator('tbody tr').count() for t in tables.all()],
            embedded_video=media,screenshots=[p.name for p in PREVIEW.glob('*.png')],
            agent_visual_review_pending=True)
        browser.close()
    (ROOT/'docs/turbovla_multitask/notebook_render.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
    print('NOTEBOOK_RENDERED',json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':inspect()
