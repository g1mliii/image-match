export interface ElectronAPI {
  selectFile: () => Promise<string[]>;
  selectFiles: () => Promise<string[]>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
