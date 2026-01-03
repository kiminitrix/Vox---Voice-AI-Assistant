
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { GoogleGenAI, LiveServerMessage } from '@google/genai';
import { ConnectionStatus, Modality } from './types';
import { decode, decodeAudioData, createPcmBlob } from './services/audioUtils';

// Global constants for the Live API
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const SAMPLE_RATE_IN = 16000;
const SAMPLE_RATE_OUT = 24000;

const App: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [intensity, setIntensity] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  // Audio Context References
  const inputAudioCtxRef = useRef<AudioContext | null>(null);
  const outputAudioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTimeRef = useRef<number>(0);
  const sessionRef = useRef<any>(null);
  const isMutedRef = useRef<boolean>(false);

  // Analysis for visual feedback
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);

  const cleanup = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (sessionRef.current) {
      sessionRef.current.close?.();
      sessionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    sourcesRef.current.forEach(source => source.stop());
    sourcesRef.current.clear();
    
    if (inputAudioCtxRef.current) inputAudioCtxRef.current.close();
    if (outputAudioCtxRef.current) outputAudioCtxRef.current.close();
    
    inputAudioCtxRef.current = null;
    outputAudioCtxRef.current = null;
    setStatus('disconnected');
    setIsSpeaking(false);
    setIntensity(0);
    setIsMuted(false);
    isMutedRef.current = false;
  }, []);

  const startAnalysis = (ctx: AudioContext, stream: MediaStream) => {
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 64;
    source.connect(analyser);
    analyserRef.current = analyser;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const update = () => {
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const average = sum / bufferLength;
      // If muted, we effectively visually zero out user input intensity, 
      // but we might still want to see some subtle pulse if the model is speaking.
      const displayedIntensity = isMutedRef.current ? (isSpeaking ? average / 512 : 0) : average / 255;
      setIntensity(displayedIntensity); 
      rafRef.current = requestAnimationFrame(update);
    };
    update();
  };

  const connect = async () => {
    try {
      setStatus('connecting');
      setErrorMessage(null);

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

      // Initialize audio contexts
      inputAudioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE_IN });
      outputAudioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE_OUT });
      const outputNode = outputAudioCtxRef.current.createGain();
      outputNode.connect(outputAudioCtxRef.current.destination);

      // Get microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      startAnalysis(inputAudioCtxRef.current, stream);

      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction: `You are a voice-only conversational AI assistant. 
          CORE RULES (ABSOLUTE):
          1. You MUST communicate ONLY through spoken language.
          2. You MUST NEVER output text, markdown, bullet points, lists, emojis, symbols, or formatting of any kind.
          3. Your responses are intended to be converted directly into audio.
          4. Speak naturally, as a human would in a real conversation.
          5. If a response sounds like writing, rephrase it into spoken language.
          
          VOICE & DELIVERY STYLE:
          - Speak in a calm, warm, clear, and confident voice.
          - Use natural conversational rhythm, including short pauses.
          - Keep sentences concise and easy to follow.
          
          LANGUAGE HANDLING:
          - Match the user’s spoken language automatically.
          - If the user speaks Malay, respond in Malay.
          - If the user speaks English, respond in English.
          - If the user mixes languages, respond naturally using the same mix.
          
          DEFAULT RESPONSE LENGTH:
          - One to three spoken sentences.
          `,
        },
        callbacks: {
          onopen: () => {
            setStatus('connected');
            
            // Start streaming audio to the model
            const source = inputAudioCtxRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioCtxRef.current!.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              // Check the ref to see if we should send audio
              if (isMutedRef.current) return;

              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createPcmBlob(inputData);
              sessionPromise.then(session => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioCtxRef.current!.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
              setIsSpeaking(true);
              const ctx = outputAudioCtxRef.current!;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              
              const audioBuffer = await decodeAudioData(
                decode(audioData),
                ctx,
                SAMPLE_RATE_OUT,
                1
              );
              
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNode);
              source.addEventListener('ended', () => {
                sourcesRef.current.delete(source);
                if (sourcesRef.current.size === 0) {
                  setIsSpeaking(false);
                }
              });
              
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => s.stop());
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setIsSpeaking(false);
            }
          },
          onerror: (e) => {
            console.error('Gemini Live Error:', e);
            setErrorMessage('A connection error occurred. Please try again.');
            cleanup();
          },
          onclose: () => {
            console.log('Gemini Live Closed');
            cleanup();
          },
        },
      });

      sessionRef.current = await sessionPromise;
    } catch (err: any) {
      console.error('Failed to connect:', err);
      setErrorMessage(err.message || 'Failed to start the conversation.');
      cleanup();
    }
  };

  const toggleConnection = () => {
    if (status === 'connected') {
      cleanup();
    } else {
      connect();
    }
  };

  const toggleMute = () => {
    const newState = !isMuted;
    setIsMuted(newState);
    isMutedRef.current = newState;
  };

  // Handle unmount cleanup
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center relative overflow-hidden bg-slate-950 px-6">
      
      {/* Background Glow Effect */}
      <div 
        className={`absolute glow-orb w-[600px] h-[600px] rounded-full transition-all duration-700 ease-in-out opacity-30 ${
          status === 'connected' 
            ? isMuted
              ? 'bg-slate-700 scale-95 blur-[120px]'
              : isSpeaking 
                ? 'bg-blue-400 scale-110 blur-[120px]' 
                : 'bg-indigo-600 scale-100 blur-[100px]'
            : status === 'connecting'
              ? 'bg-amber-400 scale-90'
              : 'bg-slate-800 scale-95'
        }`}
        style={{
          transform: `scale(${1 + intensity * 0.5})`
        }}
      />

      {/* Main Content */}
      <div className="z-10 flex flex-col items-center text-center space-y-12">
        <header>
          <h1 className="text-4xl md:text-5xl font-light tracking-tight text-white mb-2">Vox</h1>
          <p className="text-slate-400 font-light text-lg">Pure Voice Interaction</p>
        </header>

        {/* Visualizer Orb */}
        <div className="relative group">
          <div 
            className={`w-48 h-48 rounded-full flex items-center justify-center border transition-all duration-500 ${
              status === 'connected' 
                ? isMuted
                  ? 'border-slate-700 shadow-none'
                  : 'border-indigo-400 shadow-[0_0_50px_rgba(129,140,248,0.3)]' 
                : 'border-slate-800'
            }`}
            style={{
              padding: `${20 - (intensity * 10)}px`
            }}
          >
            <div 
              className={`w-full h-full rounded-full transition-all duration-300 relative flex items-center justify-center overflow-hidden ${
                status === 'connected' 
                  ? isMuted
                    ? 'bg-slate-800'
                    : isSpeaking ? 'bg-indigo-400' : 'bg-indigo-500' 
                  : status === 'connecting' ? 'bg-amber-500 animate-pulse' : 'bg-slate-800'
              }`}
              style={{
                transform: `scale(${0.8 + intensity * 0.4})`
              }}
            >
              {isMuted && (
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-slate-500">
                  <line x1="1" y1="1" x2="23" y2="23"></line>
                  <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path>
                  <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path>
                  <line x1="12" y1="19" x2="12" y2="23"></line>
                  <line x1="8" y1="23" x2="16" y2="23"></line>
                </svg>
              )}
            </div>
          </div>
          
          {/* Status Badge */}
          <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap">
            <span className={`px-4 py-1 rounded-full text-xs font-medium tracking-widest uppercase border transition-colors duration-300 ${
              status === 'connected' 
                ? isMuted 
                  ? 'bg-slate-800 text-slate-500 border-slate-700'
                  : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                : status === 'connecting'
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                : 'bg-slate-800/50 text-slate-500 border-slate-700'
            }`}>
              {status === 'connected' && isMuted ? 'Muted' : status}
            </span>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-col items-center space-y-6 w-full">
          <div className="flex items-center space-x-4">
            {status === 'connected' && (
              <button
                onClick={toggleMute}
                className={`p-4 rounded-full transition-all border ${
                  isMuted 
                    ? 'bg-red-500/10 text-red-400 border-red-500/20' 
                    : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
                }`}
                title={isMuted ? "Unmute" : "Mute"}
              >
                {isMuted ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="1" y1="1" x2="23" y2="23"></line><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                )}
              </button>
            )}
            
            <button
              onClick={toggleConnection}
              disabled={status === 'connecting'}
              className={`px-10 py-4 rounded-full font-medium transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed ${
                status === 'connected'
                  ? 'bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20'
                  : 'bg-white text-slate-950 hover:bg-slate-100 shadow-xl'
              }`}
            >
              {status === 'connected' ? 'End Conversation' : status === 'connecting' ? 'Connecting...' : 'Start Conversation'}
            </button>
          </div>
          
          {errorMessage && (
            <p className="text-red-400 text-sm max-w-md bg-red-400/10 border border-red-400/20 rounded-lg px-4 py-2">
              {errorMessage}
            </p>
          )}
        </div>
      </div>

      {/* Footer Info */}
      <footer className="absolute bottom-8 w-full px-12 flex justify-between items-center text-slate-500 text-sm">
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${status === 'connected' ? (isMuted ? 'bg-slate-700' : 'bg-emerald-500 animate-pulse') : 'bg-slate-700'}`} />
          <span>Gemini 2.5 Flash Live</span>
        </div>
        <div className="flex space-x-8 uppercase tracking-widest text-[10px]">
          <span>{isMuted ? 'Input Muted' : 'Mic Active'}</span>
          <span>Zero Latency</span>
          <span>Adaptive Voice</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
