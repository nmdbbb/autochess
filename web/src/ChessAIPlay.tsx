
"use client"

import { useState, useEffect } from "react"
import { Chess } from "chess.js"
import { Chessboard } from "react-chessboard"

export default function ChessAIPlay() {
  const [game, setGame] = useState(new Chess())
  const [fen, setFen] = useState(game.fen())
  const [thinking, setThinking] = useState(false)
  const [status, setStatus] = useState("Trận chưa bắt đầu")

  useEffect(() => {
    updateStatus()
  }, [game])

  const updateStatus = () => {
    if (game.isGameOver()) {
      if (game.isCheckmate()) {
        setStatus("Chiếu hết! " + (game.turn() === "w" ? "Đen thắng" : "Trắng thắng"))
      } else {
        setStatus("Hòa cờ")
      }
    } else {
      setStatus(game.turn() === "w" ? "Lượt trắng" : "Lượt đen")
    }
  }

  const makeAIMove = async () => {
    if (game.isGameOver()) return
    setThinking(true)
    try {
      const res = await fetch("http://localhost:5000/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen }),
      })
      const data = await res.json()
      game.move(data.move)
      const newFen = game.fen()
      setFen(newFen)
      setGame(new Chess(newFen))
      updateStatus()
    } catch (err) {
      console.error("Lỗi khi gọi API:", err)
    }
    setThinking(false)
  }

  const onDrop = (source: string, target: string) => {
    const move = game.move({
      from: source,
      to: target,
      promotion: "q",
    })

    if (move == null) return false
    const newFen = game.fen()
    setFen(newFen)
    setGame(new Chess(newFen))
    updateStatus()
    return true
  }

  const resetGame = () => {
    const newGame = new Chess()
    setGame(newGame)
    setFen(newGame.fen())
    setStatus("Trận mới bắt đầu")
  }

  return (
    <div className="flex flex-col items-center p-6 space-y-4 text-center">
      <h1 className="text-2xl font-bold text-gray-800">♟️ AlphaZero Chess AI</h1>
      <p className="text-sm text-gray-600">{status}</p>

      <div className="max-w-sm w-full">
        <Chessboard
          position={fen}
          onPieceDrop={onDrop}
          boardWidth={400}
          customBoardStyle={{ borderRadius: "8px", boxShadow: "0 4px 14px rgba(0,0,0,0.1)" }}
        />
      </div>

      <div className="flex gap-4">
        <button
          onClick={makeAIMove}
          disabled={thinking}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {thinking ? "AI đang nghĩ..." : "Cho AI đi"}
        </button>
        <button
          onClick={resetGame}
          className="bg-gray-400 hover:bg-gray-500 text-white px-4 py-2 rounded"
        >
          Chơi lại
        </button>
      </div>
    </div>
  )
}
