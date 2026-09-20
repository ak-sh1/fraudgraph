'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import * as d3 from 'd3'

interface Alert {
  txn_id: string
  timestamp: string
  amount: number
  account_id: string
  merchant_id: string
  device_id: string
  channel: string
  fraud_score: number
  is_fraud: boolean
  fraud_pattern?: string
  reasons: Array<{feature: string, value: string}>
  status: string
}

interface GraphNode {
  id: string
  type: string
  label: string
  is_center: boolean
}

interface GraphEdge {
  source: string
  target: string
  type: string
}

interface EgoGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export default function AlertDetail() {
  const params = useParams()
  const router = useRouter()
  const txnId = params.id as string
  
  const [alertData, setAlertData] = useState<Alert | null>(null)
  const [egoGraph, setEgoGraph] = useState<EgoGraph | null>(null)
  const [loading, setLoading] = useState(true)
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    Promise.all([
      fetch('/data/alerts.json').then(res => res.json()),
      fetch('/data/ego_graphs.json').then(res => res.json())
    ]).then(([alerts, graphs]) => {
      const foundAlert = alerts.find((a: Alert) => a.txn_id === txnId)
      setAlertData(foundAlert)
      setEgoGraph(graphs[txnId] || null)
      setLoading(false)
    }).catch(err => {
      console.error('Failed to load data:', err)
      setLoading(false)
    })
  }, [txnId])

  useEffect(() => {
    if (egoGraph && svgRef.current) {
      drawGraph(egoGraph, svgRef.current)
    }
  }, [egoGraph])

  const drawGraph = (graph: EgoGraph, svg: SVGSVGElement) => {
    const width = 800
    const height = 500
    
    d3.select(svg).selectAll('*').remove()
    
    const svgSelection = d3.select(svg)
      .attr('width', width)
      .attr('height', height)
    
    const g = svgSelection.append('g')
    
    const simulation = d3.forceSimulation(graph.nodes as any)
      .force('link', d3.forceLink(graph.edges).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40))
    
    const links = g.append('g')
      .selectAll('line')
      .data(graph.edges)
      .enter().append('line')
      .attr('stroke', '#95a5a6')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)
    
    const nodes = g.append('g')
      .selectAll('g')
      .data(graph.nodes)
      .enter().append('g')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any)
    
    nodes.append('circle')
      .attr('r', (d: any) => d.is_center ? 20 : 15)
      .attr('fill', (d: any) => {
        if (d.type === 'account') return d.is_center ? '#e74c3c' : '#3498db'
        if (d.type === 'merchant') return '#27ae60'
        if (d.type === 'device') return '#f39c12'
        return '#95a5a6'
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
    
    nodes.append('text')
      .text((d: any) => d.label.substring(0, 8))
      .attr('font-size', '10px')
      .attr('fill', '#2c3e50')
      .attr('text-anchor', 'middle')
      .attr('dy', 30)
    
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)
      
      nodes.attr('transform', (d: any) => `translate(${d.x},${d.y})`)
    })
    
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      d.fx = d.x
      d.fy = d.y
    }
    
    function dragged(event: any, d: any) {
      d.fx = event.x
      d.fy = event.y
    }
    
    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null
      d.fy = null
    }
  }

  const handleDisposition = async (disposition: 'confirmed_fraud' | 'false_positive') => {
    // In a real app, this would call an API
    alert(`Marked as: ${disposition}`)
    router.push('/')
  }

  if (loading || !alertData) {
    return <div className="container"><div className="header"><h1>Loading...</h1></div></div>
  }

  return (
    <div className="container">
      <div className="detail-container">
        <div className="detail-header">
          <h1>Transaction Detail: {alertData.txn_id}</h1>
          <p style={{color: '#7f8c8d', marginTop: '5px'}}>
            Fraud Score: <span className={`score-badge ${alertData.fraud_score >= 0.7 ? 'score-high' : alertData.fraud_score >= 0.4 ? 'score-medium' : 'score-low'}`}>
              {(alertData.fraud_score * 100).toFixed(1)}%
            </span>
          </p>
        </div>

        <div className="detail-section">
          <h2>Transaction Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <div className="info-label">Amount</div>
              <div className="info-value">${alertData.amount.toFixed(2)}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Account</div>
              <div className="info-value">{alertData.account_id}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Merchant</div>
              <div className="info-value">{alertData.merchant_id}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Device</div>
              <div className="info-value">{alertData.device_id}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Channel</div>
              <div className="info-value">{alertData.channel}</div>
            </div>
            <div className="info-item">
              <div className="info-label">Timestamp</div>
              <div className="info-value">{new Date(alertData.timestamp).toLocaleString()}</div>
            </div>
          </div>
        </div>

        <div className="detail-section">
          <h2>Top Fraud Reasons</h2>
          <ul className="reason-list">
            {alertData.reasons.map((reason, idx) => (
              <li key={idx} className="reason-item">
                <span className="reason-feature">{reason.feature}:</span>
                <span className="reason-value">{reason.value}</span>
              </li>
            ))}
          </ul>
        </div>

        {egoGraph && (
          <div className="detail-section">
            <h2>Ego-Graph Visualization</h2>
            <div className="graph-container">
              <svg ref={svgRef}></svg>
              <div className="graph-legend">
                <div className="legend-item">
                  <div className="legend-dot" style={{background: '#e74c3c'}}></div>
                  <span>Center Account</span>
                </div>
                <div className="legend-item">
                  <div className="legend-dot" style={{background: '#3498db'}}></div>
                  <span>Related Accounts</span>
                </div>
                <div className="legend-item">
                  <div className="legend-dot" style={{background: '#27ae60'}}></div>
                  <span>Merchants</span>
                </div>
                <div className="legend-item">
                  <div className="legend-dot" style={{background: '#f39c12'}}></div>
                  <span>Devices</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="detail-section">
          <h2>Disposition</h2>
          <div className="disposition-buttons">
            <button className="btn btn-fraud" onClick={() => handleDisposition('confirmed_fraud')}>
              ⚠️ Confirm Fraud
            </button>
            <button className="btn btn-fp" onClick={() => handleDisposition('false_positive')}>
              ✓ Mark as False Positive
            </button>
            <button className="btn btn-back" onClick={() => router.push('/')}>
              ← Back to Queue
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
