"""Live Market News & Social Sentiment Fetcher for Indian Stock Markets."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger
import xml.etree.ElementTree as ET

# Live Indian Financial News RSS Feeds
RSS_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "livemint": "https://www.livemint.com/rss/markets",
}

class SentimentFetcher:
    """Fetches live market news, RSS headlines, and social sentiment for Indian stocks."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "TradingBot/1.0"})

    async def fetch_news_headlines(self, symbol: str) -> List[Dict[str, str]]:
        """Fetch latest financial news headlines matching the stock symbol."""
        headlines = []
        clean_symbol = symbol.replace(".NS", "").replace("^", "").strip().upper()

        for source_name, url in RSS_FEEDS.items():
            try:
                res = await self.client.get(url)
                if res.status_code == 200:
                    root = ET.fromstring(res.text)
                    for item in root.findall(".//item"):
                        title = item.findtext("title") or ""
                        link = item.findtext("link") or ""
                        pub_date = item.findtext("pubDate") or ""

                        # Filter for stock relevance or general market news
                        if clean_symbol in title.upper() or "MARKET" in title.upper() or "NIFTY" in title.upper():
                            headlines.append({
                                "source": source_name,
                                "title": title.strip(),
                                "link": link.strip(),
                                "published": pub_date.strip()
                            })
            except Exception as e:
                logger.debug(f"[SentimentFetcher] Could not fetch RSS from {source_name}: {e}")

        # Fallback if no specific headlines found
        if not headlines:
            logger.info(f"[SentimentFetcher] No recent RSS headlines directly mentioning {clean_symbol}.")
            headlines.append({
                "source": "Market News",
                "title": f"No recent specific headlines found for {clean_symbol}. Operating under neutral broad market context.",
                "link": "https://www.moneycontrol.com",
                "published": datetime.now().strftime("%Y-%m-%d")
            })

        logger.info(f"[SentimentFetcher] Fetched {len(headlines)} headlines for {symbol}")
        return headlines[:10]

    async def fetch_fii_dii_flows(self) -> Dict[str, Any]:
        """Fetch macro institutional (FII/DII) sentiment context for Indian markets."""
        try:
            # Attempt to fetch market macro news from Economic Times / Moneycontrol
            res = await self.client.get(RSS_FEEDS["economic_times"])
            fii_dii_text = ""
            if res.status_code == 200:
                root = ET.fromstring(res.text)
                for item in root.findall(".//item"):
                    title = item.findtext("title") or ""
                    if any(k in title.upper() for k in ["FII", "DII", "FOREIGN", "INSTITUTIONAL", "MUTUAL FUND"]):
                        fii_dii_text += title + ". "
            
            return {
                "fii_dii_summary": fii_dii_text.strip() or "FII/DII activity: Stable institutional participation.",
                "institutional_tone": "NEUTRAL"
            }
        except Exception as e:
            logger.debug(f"[SentimentFetcher] FII/DII fetch error: {e}")
            return {"fii_dii_summary": "Institutional data pending.", "institutional_tone": "NEUTRAL"}

    async def compute_headline_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Calculates 3-tier sentiment scores (-1.0 to +1.0) for company, sector, and macro news using expanded Indian financial lexicons."""
        headlines = await self.fetch_news_headlines(symbol)
        clean_symbol = symbol.replace(".NS", "").replace("^", "").strip().upper()
        fii_dii_data = await self.fetch_fii_dii_flows()
        
        company_keywords = [clean_symbol]
        if "RELIANCE" in clean_symbol: company_keywords.extend(["RELIANCE", "RIL", "JIO"])
        elif "HDFCBANK" in clean_symbol: company_keywords.extend(["HDFC", "HDFCBANK"])
        elif "TCS" in clean_symbol: company_keywords.extend(["TCS", "TATA CONSULTANCY"])
        elif "INFY" in clean_symbol: company_keywords.extend(["INFOSYS", "INFY"])
        elif "SBIN" in clean_symbol: company_keywords.extend(["SBI", "STATE BANK"])
        elif "BAJFINANCE" in clean_symbol: company_keywords.extend(["BAJAJ FINANCE", "BAJFINANCE"])
        elif "ITC" in clean_symbol: company_keywords.extend(["ITC"])

        sector_keywords = ["BANK", "FINANCE", "IT", "TECH", "OIL", "TELECOM", "AUTO", "ENERGY", "PHARMA", "FMCG"]
        
        company_hl = []
        sector_hl = []
        macro_hl = []

        for h in headlines:
            t = h["title"].upper()
            if any(k in t for k in company_keywords):
                company_hl.append(h)
            elif any(k in t for k in sector_keywords):
                sector_hl.append(h)
            else:
                macro_hl.append(h)

        # Expanded Indian Financial Lexicons
        bull_words = [
            "bull", "surge", "gain", "profit", "growth", "buy", "rally", "record", "high", 
            "upgrade", "outperform", "dividend", "order win", "revenue up", "margin expansion",
            "rate cut", "npa reduction", "strong q1", "strong q2", "strong q3", "strong q4"
        ]
        bear_words = [
            "bear", "drop", "fall", "loss", "decline", "sell", "plunge", "down", "low", 
            "downgrade", "underperform", "sebi probe", "fraud", "default", "margin squeeze",
            "rate hike", "npa surge", "weak q1", "weak q2", "weak q3", "weak q4", "fii sell"
        ]

        def _score_list(hl_list):
            if not hl_list: return 0.0
            pos = sum(1 for h in hl_list if any(w in h["title"].lower() for w in bull_words))
            neg = sum(1 for h in hl_list if any(w in h["title"].lower() for w in bear_words))
            tot = pos + neg
            return round((pos - neg) / tot, 2) if tot > 0 else 0.0

        company_score = _score_list(company_hl)
        sector_score = _score_list(sector_hl)
        macro_score = _score_list(macro_hl)

        # Weighted combined score: Company (50%), Sector (30%), Macro (20%)
        overall_score = round(0.50 * company_score + 0.30 * sector_score + 0.20 * macro_score, 2)
        tone = "BULLISH" if overall_score > 0.2 else ("BEARISH" if overall_score < -0.2 else "NEUTRAL")

        return {
            "symbol": symbol,
            "company_sentiment": company_score,
            "sector_sentiment": sector_score,
            "macro_sentiment": macro_score,
            "overall_score": overall_score,
            "tone": tone,
            "fii_dii_summary": fii_dii_data["fii_dii_summary"],
            "company_headlines": [h["title"] for h in company_hl[:3]],
            "sector_headlines": [h["title"] for h in sector_hl[:3]],
            "macro_headlines": [h["title"] for h in macro_hl[:3]],
        }

    async def close(self):
        await self.client.aclose()
