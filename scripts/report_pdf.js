// Print reports/report.html to reports/report.pdf with Node Playwright:  node scripts/report_pdf.js
const path = require('path'); const { chromium } = require('playwright');
(async () => { const root = path.resolve(__dirname, '..'); const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + path.join(root, 'reports', 'report.html'));
  await p.pdf({ path: path.join(root, 'reports', 'report.pdf'), format: 'A4', printBackground: true, preferCSSPageSize: true });
  await b.close(); })();
