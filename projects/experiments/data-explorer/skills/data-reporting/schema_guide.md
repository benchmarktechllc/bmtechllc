# Store Sales Schema & Business Logic Guide

## Tables & Columns
- **`orders`**: `id` (INTEGER), `customer_name` (TEXT), `amount` (REAL), `status` (TEXT), `order_date` (TEXT, ISO-8601 `YYYY-MM-DD`).

## Reporting Rules & Edge Cases
- **Excluded Statuses:** Always exclude orders where `status` is `'VOIDED'` or `'CANCELLED'` from revenue totals.
- **Refund Handling:** Orders marked `'REFUNDED'` should be listed in a separate line item and subtracted from net revenue.
- **Currency Formatting:** All monetary values in final reports must be rounded to two decimal places (`$X,XXX.XX`).