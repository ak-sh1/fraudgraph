'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

interface Alert {
  txn_id: string
  timestamp: string
  amount: number
  account_id: string
  merchant_id: string
  fraud_score: number
  is_fraud: boolean
  fraud_pattern?: string
  status: string
}

export default function Home() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/data/alerts.json')
      .then(res => res.json())
      .then(data => {
        setAlerts(data)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load alerts:', err)
        setLoading(false)
      })
  }, [])

  const getScoreClass = (score: number) => {
    if (score >= 0.7) return 'score-high'
    if (score >= 0.4) return 'score-medium'
    return 'score-low'
  }

  if (loading) {
    return (
      <div className="container">
        <div className="header">
          <h1>FraudGraph</h1>
          <p>Loading alerts...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container">
      <div className="header">
        <h1>🔍 FraudGraph</h1>
        <p>Transaction Fraud Detection - {alerts.length} alerts ranked by score</p>
      </div>

      <div className="alert-queue">
        <table className="alert-table">
          <thead>
            <tr>
              <th>Score</th>
              <th>Transaction ID</th>
              <th>Amount</th>
              <th>Account</th>
              <th>Merchant</th>
              <th>Timestamp</th>
              <th>Pattern</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map(alert => (
              <tr key={alert.txn_id}>
                <td>
                  <Link href={`/alerts/${alert.txn_id}`}>
                    <span className={`score-badge ${getScoreClass(alert.fraud_score)}`}>
                      {(alert.fraud_score * 100).toFixed(1)}%
                    </span>
                  </Link>
                </td>
                <td>
                  <Link href={`/alerts/${alert.txn_id}`} style={{color: '#3498db', textDecoration: 'none'}}>
                    {alert.txn_id}
                  </Link>
                </td>
                <td>${alert.amount.toFixed(2)}</td>
                <td>{alert.account_id}</td>
                <td>{alert.merchant_id}</td>
                <td>{new Date(alert.timestamp).toLocaleString()}</td>
                <td>
                  {alert.fraud_pattern && (
                    <span style={{fontSize: '12px', color: '#7f8c8d'}}>
                      {alert.fraud_pattern.replace(/_/g, ' ')}
                    </span>
                  )}
                </td>
                <td>
                  <span className={`status-badge status-${alert.status}`}>
                    {alert.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
