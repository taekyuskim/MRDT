"""Resumable, verified MVImgNet2.0 downloader.

The official download_tool.py logs in correctly but its folder listing no longer
works with the current SharePoint page, and it reads whole archives into memory to
hash them. This script reuses its password login, lists files via the SharePoint
REST API, downloads several archives in parallel with HTTP Range resume, and
verifies each archive against the cloud quickXorHash before marking it complete.

    export MVIMGNET_PASSWORD=...        # or omit to be prompted
    python mvimgnet_download.py --url <share link> --out <dir> [--files mvi2_00 mvi2_01] [--jobs 8]

Completed archives are <out>/<name>; partial ones are <name>.part; verified
archives also get <out>/.verified/<name>.json. Rerunning resumes and skips verified files.
"""
import argparse
import asyncio
import base64
import getpass
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import quickxorhash
import requests

SITE = 'https://cuhko365.sharepoint.com/sites/GAP_Lab_MVImgNet'
FOLDER = '/sites/GAP_Lab_MVImgNet/Shared Documents/MVImgNet2.0/MVImgNet2.0_part1'
DRIVE_PATH = 'MVImgNet2.0/MVImgNet2.0_part1'
CHUNK = 8 << 20
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/90.0.4430.93 Safari/537.36 Edg/90.0.818.51')


async def browser_login(url, password):
    """Same password-page flow as the official download_tool.py, returning a cookie header."""
    from pyppeteer import launch
    # No signal handlers: re-login runs in worker threads, where signal.signal() is not allowed.
    browser = await launch(options={'args': ['--no-sandbox'], 'handleSIGINT': False,
                                    'handleSIGTERM': False, 'handleSIGHUP': False})
    try:
        page = await browser.newPage()
        await page.goto(url, {'waitUntil': 'networkidle0'})
        await page.focus("input[id='txtPassword']")
        await page.keyboard.type(password)
        button = await page.querySelector("input[id='btnSubmitPassword']")
        await asyncio.gather(page.waitForNavigation(), button.click())
        cookies = await page.cookies()
    finally:
        await browser.close()
    return ''.join(f"{c['name']}={c['value']};" for c in cookies)


class Session:
    """Cookie session from the share-link password page; re-login is serialized."""

    def __init__(self, url, password):
        self.url, self.password = url, password
        self.lock = threading.Lock()
        self.generation = 0
        self.login()

    def login(self):
        loop = asyncio.new_event_loop()
        try:
            cookie = loop.run_until_complete(browser_login(self.url, self.password))
        finally:
            loop.close()
        if not cookie:
            raise RuntimeError('Login failed: no cookies (check link/password)')
        http = requests.Session()
        http.headers.update({'cookie': cookie, 'accept': 'application/json', 'user-agent': USER_AGENT})
        self.http = http
        self.generation += 1

    def relogin(self, seen_generation):
        with self.lock:
            if self.generation == seen_generation:  # nobody refreshed it meanwhile
                for attempt in range(1, 11):
                    log(f're-login (attempt {attempt})')
                    try:
                        self.login()
                        return
                    except Exception as error:
                        log(f're-login failed: {type(error).__name__}: {error}')
                        time.sleep(min(600, 30 * attempt))
                raise RuntimeError('Re-login kept failing')

    def get_json(self, url):
        for attempt in range(5):
            generation = self.generation
            try:
                r = self.http.get(url, timeout=60)
                if r.status_code in (401, 403):
                    self.relogin(generation)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as error:
                log(f'metadata retry {attempt + 1}: {error}')
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError('Metadata request kept failing: ' + url)


LOG_LOCK = threading.Lock()


def log(message):
    with LOG_LOCK:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def list_archives(session):
    data = session.get_json(f"{SITE}/_api/web/GetFolderByServerRelativePath(decodedurl='{quote(FOLDER)}')/Files")
    return sorted((f['Name'], int(f['Length'])) for f in data['value'])


def file_qxh(path):
    h = quickxorhash.quickxorhash()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode('ascii')


def download(session, name, out, max_attempts):
    final, part, marker = out / name, out / (name + '.part'), out / '.verified' / (name + '.json')
    # A verified marker without the archive means it was extracted and deleted downstream.
    if marker.exists() and (not final.exists() or final.stat().st_size == json.loads(marker.read_text())['bytes']):
        log(f'{name}: already verified, skip')
        return name, 'skipped'
    for attempt in range(1, max_attempts + 1):
        try:
            meta = session.get_json(f'{SITE}/_api/v2.1/drive/root:/{DRIVE_PATH}/{quote(name)}')
        except Exception as error:  # never let one file's metadata failure stop the others
            log(f'{name}: attempt {attempt} metadata failed: {type(error).__name__}: {error}')
            time.sleep(min(300, 10 * attempt))
            continue
        size, expected, url = meta['size'], meta['file']['hashes']['quickXorHash'], meta['@content.downloadUrl']
        if final.exists() and not marker.exists():
            final.rename(part)  # unverified leftover: verify/resume it as a part
        done = part.stat().st_size if part.exists() else 0
        if done > size:
            part.unlink(); done = 0
        try:
            if done < size:
                r = requests.get(url, headers={'Range': f'bytes={done}-'}, stream=True, timeout=120)
                if r.status_code not in (200, 206):
                    raise requests.HTTPError(f'HTTP {r.status_code}')
                if r.status_code == 200 and done:
                    log(f'{name}: server ignored Range, restarting from 0'); done = 0
                with open(part, 'ab' if done else 'wb') as f:
                    last = time.time(); mark = done
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk); done += len(chunk)
                        if time.time() - last > 300:
                            log(f'{name}: {done / size:6.1%} ({done / 1e9:.1f}/{size / 1e9:.1f} GB, '
                                f'{(done - mark) / (time.time() - last) / 1e6:.1f} MB/s)')
                            last = time.time(); mark = done
            if done != size:
                raise IOError(f'incomplete: {done}/{size}')
            log(f'{name}: download complete, verifying quickXorHash')
            actual = file_qxh(part)
            if actual != expected:
                log(f'{name}: HASH MISMATCH ({actual} != {expected}); deleting and retrying')
                part.unlink()
                continue
            os.replace(part, final)
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps({'name': name, 'bytes': size, 'quickXorHash': expected,
                                          'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S')}))
            log(f'{name}: verified OK')
            return name, 'verified'
        except (requests.RequestException, IOError) as error:
            log(f'{name}: attempt {attempt} interrupted at {done / 1e9:.1f} GB: {error}')
            time.sleep(min(300, 10 * attempt))
    return name, 'failed'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', required=True, help='Share link from the MVImgNet2.0 access email')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--files', nargs='*', help='Subset, e.g. mvi2_00 mvi2_01 (default: all)')
    p.add_argument('--jobs', type=int, default=8, help='Archives downloaded in parallel')
    p.add_argument('--max-attempts', type=int, default=50)
    p.add_argument('--list-only', action='store_true')
    args = p.parse_args()
    password = os.environ.get('MVIMGNET_PASSWORD') or getpass.getpass('MVImgNet2.0 password: ')
    session = Session(args.url, password)
    archives = list_archives(session)
    if args.files:
        wanted = {f if f.endswith('.tar.gz') else f + '.tar.gz' for f in args.files}
        missing = wanted - {n for n, _ in archives}
        if missing:
            p.error(f'not in share: {sorted(missing)}')
        archives = [a for a in archives if a[0] in wanted]
    total = sum(s for _, s in archives)
    log(f'{len(archives)} archives, {total / 1e12:.3f} TB')
    if args.list_only:
        for n, s in archives:
            print(f'{s / 1e9:8.2f} GB  {n}')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(args.out).f_bavail * os.statvfs(args.out).f_frsize
    have = sum((args.out / n).stat().st_size if (args.out / n).exists() else
               (args.out / (n + '.part')).stat().st_size if (args.out / (n + '.part')).exists() else 0
               for n, _ in archives)
    if total - have > free:
        sys.exit(f'Not enough space: need {(total - have) / 1e12:.2f} TB more, {free / 1e12:.2f} TB free')
    with ThreadPoolExecutor(args.jobs) as pool:
        results = dict(f.result() for f in as_completed(
            pool.submit(download, session, n, args.out, args.max_attempts) for n, _ in archives))
    failed = sorted(n for n, r in results.items() if r == 'failed')
    log(f'summary: {sum(r != "failed" for r in results.values())}/{len(results)} verified; failed={failed}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
