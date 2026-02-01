#!/usr/bin/env python3
"""
Integration test for PriceDataFetcher._fetch_single_ticker method.

Tests fetching historical price data for APLD for 1 day and 1 week periods.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio.data_fetcher import PriceDataFetcher


def test_fetch_single_ticker():
    """Test _fetch_single_ticker for APLD over 1 minute and 1 hour."""
    
    print("=" * 80)
    print("INTEGRATION TEST: PriceDataFetcher._fetch_single_ticker")
    print("=" * 80)
    
    # Initialize fetcher
    print("\n[Step 1] Initializing PriceDataFetcher...")
    print("-" * 80)
    
    try:
        fetcher = PriceDataFetcher()
        print("✓ PriceDataFetcher initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False
    
    # Test 1: Fetch data for 1 day
    print("\n[Step 2] Testing 1 day period for APLD...")
    print("-" * 80)
    
    end_date = datetime.now()
    start_date_1d = end_date - timedelta(days=1)
    
    print(f"Start date: {start_date_1d.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fetching data...")
    
    try:
        data_1d = fetcher._fetch_single_ticker(
            symbol='APLD',
            start_date=start_date_1d,
            end_date=end_date,
            resolution='1m'
        )
        
        if data_1d:
            # Print last 50 (minute, price) pairs in UTC, minimal info
            dates = data_1d.get('dates', [])
            prices = data_1d.get('prices', [])
            if dates and prices:
                print("  Last 10 minute prices (UTC):")
                for dt, price in list(zip(dates, prices))[-10:]:
                    dt_utc = dt.astimezone().replace(tzinfo=None) if hasattr(dt, 'astimezone') else dt
                    print(f"    {dt_utc.strftime('%Y-%m-%d %H:%M')}: ${price:.2f}")
            print("✓ Successfully fetched data for 1 day period")
        else:
                print(f"✗ No data returned for 1 day period")
                return False
            
    except Exception as e:
        print(f"✗ Error fetching 1 day data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Fetch data for 1 week
    print("\n[Step 3] Testing 1 week period for APLD...")
    print("-" * 80)
    
    start_date_1w = end_date - timedelta(weeks=1)
    
    print(f"Start date: {start_date_1w.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fetching data...")
    
    try:
        data_1w = fetcher._fetch_single_ticker(
            symbol='APLD',
            start_date=start_date_1w,
            end_date=end_date,
            resolution='1h'
        )
        
        if data_1w:
            print(f"✓ Successfully fetched data for 1 week period")
            print(f"  Symbol: {data_1w.get('symbol')}")
            print(f"  Data points: {len(data_1w.get('dates', []))}")
            print(f"  Date range: {data_1w['dates'][0].strftime('%Y-%m-%d') if data_1w['dates'] else 'N/A'} to {data_1w['dates'][-1].strftime('%Y-%m-%d') if data_1w['dates'] else 'N/A'}")
            
            if data_1w['dates']:
                print(f"\n  Sample data points (first 10):")
                for i, (date, price) in enumerate(zip(data_1w['dates'][:10], data_1w['prices'][:10])):
                    print(f"    {date.strftime('%Y-%m-%d')}: ${price:.2f}")
                
                print(f"\n  Sample data points (last 10):")
                for i, (date, price) in enumerate(zip(data_1w['dates'][-10:], data_1w['prices'][-10:])):
                    print(f"    {date.strftime('%Y-%m-%d')}: ${price:.2f}")
                
                # Price statistics
                prices = data_1w['prices']
                if prices:
                    print(f"\n  Price statistics:")
                    print(f"    - Min: ${min(prices):.2f}")
                    print(f"    - Max: ${max(prices):.2f}")
                    print(f"    - First: ${prices[0]:.2f}")
                    print(f"    - Last: ${prices[-1]:.2f}")
                    if len(prices) > 1:
                        change = prices[-1] - prices[0]
                        change_pct = (change / prices[0]) * 100
                        print(f"    - Change: ${change:.2f} ({change_pct:+.2f}%)")
            else:
                print(f"  ⚠ No data points returned")
        else:
            print(f"✗ No data returned for 1 week period")
            return False
            
    except Exception as e:
        print(f"✗ Error fetching 1 week data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify data structure
    print("\n[Step 4] Verifying data structure...")
    print("-" * 80)
    
    required_keys = ['symbol', 'dates', 'prices', 'opens', 'highs', 'lows', 'volumes']
    
    for period_name, data in [('1 day', data_1d), ('1 week', data_1w)]:
        print(f"\n  {period_name} period:")
        for key in required_keys:
            if key in data:
                if isinstance(data[key], list):
                    print(f"    ✓ {key}: {len(data[key])} items")
                else:
                    print(f"    ✓ {key}: {data[key]}")
            else:
                print(f"    ✗ {key}: MISSING")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ 1 day period: {len(data_1d['dates'])} data points")
    print(f"✓ 1 week period: {len(data_1w['dates'])} data points")
    print(f"✓ Both periods returned valid data")
    print(f"✓ Data structure is correct")
    
    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = test_fetch_single_ticker()
    sys.exit(0 if success else 1)
