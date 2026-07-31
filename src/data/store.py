import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from loguru import logger

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Float, DateTime, Enum as SQLEnum, select, update

from src.core.models import Candle, Signal, Order, Trade
from src.core.enums import TimeFrame, Exchange, SignalAction, OrderStatus, Side, OrderType

Base = declarative_base()

class DBCandle(Base):
    __tablename__ = 'candles'
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    exchange: Mapped[Exchange] = mapped_column(SQLEnum(Exchange))
    timeframe: Mapped[TimeFrame] = mapped_column(SQLEnum(TimeFrame), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

class DBSignal(Base):
    __tablename__ = 'signals'
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    action: Mapped[SignalAction] = mapped_column(SQLEnum(SignalAction))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)

class DBOrder(Base):
    __tablename__ = 'orders'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[Side] = mapped_column(SQLEnum(Side))
    order_type: Mapped[OrderType] = mapped_column(SQLEnum(OrderType))
    status: Mapped[OrderStatus] = mapped_column(SQLEnum(OrderStatus))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

class DBTrade(Base):
    __tablename__ = 'trades'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    order_id: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[Side] = mapped_column(SQLEnum(Side))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

class DataStore:
    """
    SQLite database layer for storing market data, signals, orders, and trades.
    Uses SQLAlchemy async ORM.
    """
    def __init__(self, db_path: str = "sqlite+aiosqlite:///data/trading.db"):
        if "sqlite" in db_path:
            # Extract path from URL (e.g. sqlite+aiosqlite:///data/trading.db -> data/trading.db)
            path_str = db_path.split(":///")[-1]
            if path_str != ":memory:":
                Path(path_str).parent.mkdir(parents=True, exist_ok=True)
                
        self.engine = create_async_engine(db_path, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schemas initialized.")

    async def save_candles(self, candles: list[Candle]) -> None:
        """Upsert a list of candles."""
        async with self.async_session() as session:
            async with session.begin():
                for c in candles:
                    # In a real high-throughput system, use bulk insert or sqlite ON CONFLICT
                    # Simple add for now
                    db_candle = DBCandle(
                        symbol=c.symbol,
                        exchange=c.exchange,
                        timeframe=c.timeframe,
                        timestamp=c.timestamp,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume
                    )
                    session.add(db_candle)
            await session.commit()

    async def get_candles(self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime) -> list[Candle]:
        """Fetch candles for a symbol, timeframe, and date range."""
        async with self.async_session() as session:
            stmt = select(DBCandle).where(
                DBCandle.symbol == symbol,
                DBCandle.timeframe == timeframe,
                DBCandle.timestamp >= start,
                DBCandle.timestamp <= end
            ).order_by(DBCandle.timestamp.asc())
            
            result = await session.execute(stmt)
            db_candles = result.scalars().all()
            
            return [
                Candle(
                    symbol=c.symbol,
                    exchange=c.exchange,
                    timeframe=c.timeframe,
                    timestamp=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume
                ) for c in db_candles
            ]

    async def save_signal(self, signal: Signal) -> None:
        """Save a generated trading signal."""
        async with self.async_session() as session:
            async with session.begin():
                db_sig = DBSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    timestamp=signal.timestamp,
                    confidence=getattr(signal, 'confidence', 0.0)
                )
                session.add(db_sig)
            await session.commit()

    async def save_order(self, order: Order) -> None:
        """Save a new order."""
        async with self.async_session() as session:
            async with session.begin():
                db_order = DBOrder(
                    id=getattr(order, 'id', str(uuid.uuid4())),
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    status=order.status,
                    quantity=order.quantity,
                    price=order.price,
                    timestamp=order.timestamp
                )
                session.add(db_order)
            await session.commit()

    async def update_order(self, order: Order) -> None:
        """Update an existing order's status and details."""
        async with self.async_session() as session:
            async with session.begin():
                order_id = getattr(order, 'id', None)
                if not order_id:
                    logger.error("Cannot update order without an ID")
                    return
                    
                stmt = update(DBOrder).where(DBOrder.id == order_id).values(
                    status=order.status,
                    quantity=order.quantity,
                    price=order.price
                )
                await session.execute(stmt)
            await session.commit()

    async def save_trade(self, trade: Trade) -> None:
        """Save a completed trade."""
        async with self.async_session() as session:
            async with session.begin():
                db_trade = DBTrade(
                    id=getattr(trade, 'id', str(uuid.uuid4())),
                    order_id=trade.order_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=trade.quantity,
                    price=trade.price,
                    timestamp=trade.timestamp
                )
                session.add(db_trade)
            await session.commit()

    async def get_trades(self, symbol: Optional[str] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> list[Trade]:
        """Fetch trades, optionally filtered by symbol and time range."""
        async with self.async_session() as session:
            stmt = select(DBTrade)
            if symbol:
                stmt = stmt.where(DBTrade.symbol == symbol)
            if start:
                stmt = stmt.where(DBTrade.timestamp >= start)
            if end:
                stmt = stmt.where(DBTrade.timestamp <= end)
                
            stmt = stmt.order_by(DBTrade.timestamp.asc())
            result = await session.execute(stmt)
            db_trades = result.scalars().all()
            
            return [
                Trade(
                    id=t.id,
                    order_id=t.order_id,
                    symbol=t.symbol,
                    side=t.side,
                    quantity=t.quantity,
                    price=t.price,
                    timestamp=t.timestamp
                ) for t in db_trades
            ]
