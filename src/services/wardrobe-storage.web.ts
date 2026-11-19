import AsyncStorage from '@react-native-async-storage/async-storage';

// Photo data exceeds localStorage's small string quota after a few originals.
// Keep the original legacy record intact, and commit the full wardrobe atomically.
let connection: Promise<IDBDatabase> | undefined;
function database(): Promise<IDBDatabase> {
  if (!connection) {
    connection = new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('wardrobe-rag-local', 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains('wardrobe')) request.result.createObjectStore('wardrobe');
      };
      request.onerror = () => reject(request.error || new Error('사진 저장소를 열지 못했어요.'));
      request.onblocked = () => reject(new Error('다른 앱 탭을 닫고 다시 저장해주세요.'));
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => { db.close(); connection = undefined; };
        resolve(db);
      };
    }).catch(error => { connection = undefined; throw error; });
  }
  return connection;
}

export const wardrobeStorage = {
  async getItem(key: string): Promise<string | null> {
    if (typeof indexedDB === 'undefined') return AsyncStorage.getItem(key);
    const db = await database();
    const raw = await new Promise<string | null>((resolve, reject) => {
      const tx = db.transaction('wardrobe', 'readonly');
      const request = tx.objectStore('wardrobe').get(key);
      tx.oncomplete = () => resolve(typeof request.result === 'string' ? request.result : null);
      tx.onabort = () => reject(tx.error || new Error('저장한 옷장을 읽지 못했어요.'));
      tx.onerror = () => reject(tx.error || new Error('저장한 옷장을 읽지 못했어요.'));
    });
    // The provider validates this old record before allowing the first write.
    return raw ?? AsyncStorage.getItem(key);
  },
  async setItem(key: string, value: string): Promise<void> {
    if (typeof indexedDB === 'undefined') return AsyncStorage.setItem(key, value);
    const db = await database();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction('wardrobe', 'readwrite');
      tx.objectStore('wardrobe').put(value, key);
      tx.oncomplete = () => resolve();
      tx.onabort = () => reject(tx.error || new Error('옷장 저장을 완료하지 못했어요.'));
      tx.onerror = () => reject(tx.error || new Error('옷장 저장을 완료하지 못했어요.'));
    });
  },
};
