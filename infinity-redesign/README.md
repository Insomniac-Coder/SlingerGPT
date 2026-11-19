# Infinity Phone Charms: redesign concept

A clickable redesign mockup for infinityphonecharms.com. It's built from the shop's real catalog: 102 products, with their prices, stock levels and photos as of 19 Nov 2026.

## How to view

Unzip the folder and open **`index.html`** in any browser. Everything is local, so no server is needed (fonts load from Google Fonts when you're online).

| Page | What it shows |
|---|---|
| `index.html` | Overview: audit of the current site, before/after, brand system, Squarespace how-to, roadmap |
| `home.html` | New homepage |
| `shop.html` | Shop with type/vibe/price/stock filters, sorting and load more. Try `shop.html?vibe=holiday` |
| `product.html?id=…` | Product page. Click any product, or open it bare to see the Lip Balm Keychain with flavor swatches |
| `custom.html` | Custom-order bead builder + request form + group orders |
| `about.html` | About + help center (shipping, returns, fit, care, contact) |

On any page, click **Design notes** (bottom-right) to show numbered pins that explain each change.

The `previews/` folder has ready-made screenshots of every page, at desktop and mobile size.

## Placeholders to confirm

Copy with a **dotted pink underline** is a placeholder or suggestion: shipping times and cost, the $25 free-shipping threshold, returns policy, holiday cutoffs, custom pricing, group discounts, her name, photo and social handles. The review cards are a sample layout only.

## Notes

- Product names in the prototype are cleaned up (39 renamed). Each product page shows the old name.
- Nothing is wired to a real checkout, newsletter or form. Those buttons show a "Mockup" message.
- `assets/catalog.js` holds the product data and `assets/img/` holds the product photos, resized for the mockup.
