# Lead Intelligence Dashboard

## Purpose of the Application
The **Lead Intelligence Dashboard** is an enterprise-grade web application built with Dash and Bootstrap. It is designed for RevOps, leadership, and marketing teams to centralize lead evaluation, monitor multi-channel campaign pipline, and streamline sales prioritization. 

By automating lead scoring and offering interactive profile drill-downs and customer journey flow charts, the application provides a high-level view of campaign performance and suggests next actions for sales teams.

---

## Data Cleaning & Preparation Process
The data pipeline ingests raw lead data (`messy_leads_mock_dataset.csv`) and executes a multi-step cleaning, standardization, and feature engineering process prior to downstream dashboard rendering:

* **Mixed Date Parsing:** Converts `created_date`, `first_contact_date`, and `conversion_date` into a standardized `YYYY-MM-DD` date format using `dateutil.parser`, safely converting missing or malformed dates to `None`.
* **Text & String Standardization:** Strips redundant whitespace and normalizes text fields (e.g., job titles, notes, industries) to Title Case. State abbreviations are converted to uppercase (e.g., `OH`, `TX`), and email addresses are normalized to lowercase.
* **Country Name Normalization:** Maps inconsistent country inputs (`us`, `usa`, `united states`, `united states of america`) to a single standardized country label (`United Sates`).
* **Full Name Feature Engineering:** Concatenates `lead_first_name` and `lead_last_name` into a unified `lead_full_name` column.
* **Decision Maker Identification:** Performs pattern matching (`Founder|Ceo|Head|Vp|Director`) across `job_title` values to automatically populate a boolean `is_decision_maker` flag.
* **Sales Rep Assignment Flagging:** Evaluates `sales_rep` values to categorize leads into `Assigned` vs. `Unassigned` via the `is_assigned` feature.
* **Data Export:** Exports the cleaned DataFrame to both a pickled object (`lead_cleaning.pkl`) for application loading and a CSV (`data_cleaning.csv`) for auditing.

---

## Lead Distribution

### Round-Robin Assignment Model
Unassigned leads are allocated using a **Round-Robin Assignment Model**. This method was specifically chosen to ensure that unassigned leads are evenly distributed across all available team members, preventing workload imbalances and maintaining consistent lead coverage.

**How it works:**
* Leads without an assigned owner are queued for distribution.
* The algorithm sequentially assigns each lead to the next eligible representative in the cycle.
* Once all active reps have received a lead, the sequence resets to the beginning of the list.

## Lead Prioritization Model
The dashboard features an automated **Lead Priority Score** algorithm that assigns points to leads based on six core criteria:

$$\text{Priority Score} = \text{Activity Points} + \text{Lead Score Points} + \text{Seniority Points} + \text{Status Points} + \text{Notes Points} + \text{Deal Value Points}$$

### Scoring Rules Breakdown
1. **Last Activity Engagement (Up to +20 pts):**
   * Demo Requests (`+20`), Contact Forms (`+18`), Phone Calls (`+15`), Form Fills (`+12`), Content Downloads (`+10`), Email Clicks (`+8`), Website Visits (`+5`).
2. **Historical Lead Score Tier (Up to +20 pts):**
   * $\ge 90$ (`+20`), $\ge 80$ (`+15`), $\ge 70$ (`+10`), $\ge 60$ (`+8`), $\ge 50$ (`+5`), $\ge 40$ (`+2`).
3. **Job Seniority / Authority (Up to +20 pts):**
   * C-Level/Founders (`+20`), VPs (`+18`), Directors (`+15`), Senior Managers (`+12`), Managers (`+10`), Leads (`+8`).
4. **Funnel Status Readiness (Up to +15 pts):**
   * Qualified / New (`+15`), Contacted (`+10`), Nurturing (`+5`), Converted / Lost (`0`).
5. **Note Intent Signals (Up to +15 pts):**
   * High Intent ("demo", "pricing") (`+15`), Moderate Intent ("follow up", "referral") (`+10`), General Interest (`+5`).
6. **Estimated Deal Value (Up to +15 pts):**
   * $100\text{k}$ (`+15`), $75\text{k}$ (`+12`), $50\text{k}$ (`+10`), $25\text{k}$ (`+7`), $10\text{k}$ (`+5`).

### Priority Ranking (1–10 Scale)
Leads are evaluated using percentile ranking (`rank(pct=True)`) and grouped into 10 priority ranks:
* **Priority 1:** Top 10% highest-scoring leads (Immediate action for Sales Reps).
* **Priority 10:** Bottom 10% lowest-scoring active leads.

---

## Key Metrics & Definitions

| Metric | Formula / Calculation | Definition |
| :--- | :--- | :--- |
| **Total Leads** | $\text{sum}(\text{Lead ID)}$ | Count of distinct lead ids. |
| **Contact Rate** | $\left( \frac{\text{Contacted Leads}}{\text{Total Leads}} \right) \times 100$ | Percentage of total leads reached out to by sales reps. |
| **SQL Rate** | $\left( \frac{\text{Qualified Leads}}{\text{Total Leads}} \right) \times 100$ | Percentage of leads meeting qualification criteria (Sales Qualified Leads). |
| **Win Rate** | $\left( \frac{\text{Converted Leads}}{\text{Total Leads}} \right) \times 100$ | Percentage of total leads converted into closed deals. |
| **Est. Pipeline Value** | $\text{sum}(\text{Estimated Deal Value USD})$ | Total potential dollar value across active leads in the current selection. |

---

## Visualizations & Dashboard Components

### 1. Global Filter Panel & Fuzzy Search
* Dynamic header controls filtering the dataset by Company, Industry, Campaign, Status, Sales Rep, and free-text search queries.

### 2. Interactive Lead Prioritization Table
* Ranked queue of non-lost leads ordered by Priority Rank (1 to 10).
* **Links:**
  * **Lead Name Click:** Opens a **Lead Profile** with full contact info, job title, industry, location, source, and submission history.
  * **Company Name Click:** Opens a **Company Profile** displaying organization size, location, lead volume, last touchpoint date, and primary Decision Maker.

### 3. Campaign Performance Breakdown Table
* A multi-stage metric matrix detailing total leads, funnel progression (New, Contacted, SQL, Closed Won), and efficiency conversion rates per campaign.

### 4. Company Activity Matrix (Scatter Plot)
* A 2D normalized scatter plot mapping **Marketing Activity** (x-axis) vs. **Sales Activity** (y-axis) per company to identify high-intent/low-touch target accounts.

### 5. Customer Journey Funnel Analysis (Sankey Diagram)
* Multi-stage volume flows tracking lead conversion pathways from **Marketing Campaigns** $\rightarrow$ **Assignment Status** $\rightarrow$ **Final Funnel Status**.

---

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install dash dash-bootstrap-components pandas numpy plotly scikit-learn python-dateutil
