import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'FraudGraph - Transaction Fraud Detection',
  description: 'Graph-based fraud detection with analyst case queue',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
