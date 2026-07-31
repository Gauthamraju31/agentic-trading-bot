"""Indian market fee calculator."""

from src.core.config import settings
from src.core.enums import PositionType, Side
from src.core.models import OrderFees


class FeeCalculator:
    """Calculates Indian market trading fees and taxes."""

    def __init__(self) -> None:
        self.config = settings.fees

    def calculate_fees(
        self, side: Side, quantity: int, price: float, position_type: PositionType
    ) -> OrderFees:
        """Calculate fees for a single order fill.
        
        Args:
            side: Buy or sell.
            quantity: Number of shares.
            price: Fill price.
            position_type: Delivery or intraday.
            
        Returns:
            OrderFees object with breakdown.
        """
        turnover = quantity * price

        # Brokerage: min(brokerage_per_order, 0.03% of turnover)
        brokerage = min(self.config.brokerage_per_order, turnover * 0.0003)

        # STT: delivery is charged on BOTH buy and sell; intraday only on sell.
        stt = 0.0
        if position_type == PositionType.DELIVERY:
            stt = turnover * (self.config.stt_delivery_pct / 100.0)
        elif side == Side.SELL:
            stt = turnover * (self.config.stt_intraday_sell_pct / 100.0)

        # Exchange charges
        exchange_charges = turnover * (self.config.exchange_charges_pct / 100.0)

        # SEBI charges
        sebi_charges = turnover * (self.config.sebi_charges_pct / 100.0)

        # GST: 18% on (brokerage + exchange charges + SEBI charges)
        gst = (brokerage + exchange_charges + sebi_charges) * (self.config.gst_pct / 100.0)

        # Stamp duty: only on buy side; higher rate for delivery than intraday.
        stamp_duty = 0.0
        if side == Side.BUY:
            if position_type == PositionType.DELIVERY:
                stamp_duty = turnover * (self.config.stamp_duty_delivery_buy_pct / 100.0)
            else:
                stamp_duty = turnover * (self.config.stamp_duty_intraday_buy_pct / 100.0)

        return OrderFees(
            brokerage=round(brokerage, 4),
            stt=round(stt, 4),
            exchange_charges=round(exchange_charges, 4),
            gst=round(gst, 4),
            sebi_charges=round(sebi_charges, 4),
            stamp_duty=round(stamp_duty, 4),
        )

    def total_transaction_cost(
        self, buy_price: float, sell_price: float, quantity: int, position_type: PositionType
    ) -> float:
        """Calculate the total round-trip transaction costs."""
        buy_fees = self.calculate_fees(Side.BUY, quantity, buy_price, position_type)
        sell_fees = self.calculate_fees(Side.SELL, quantity, sell_price, position_type)
        return buy_fees.total + sell_fees.total
