
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface TranscriptionEntry {
  role: 'user' | 'model';
  text: string;
}

export enum Modality {
  AUDIO = 'AUDIO',
  TEXT = 'TEXT'
}
