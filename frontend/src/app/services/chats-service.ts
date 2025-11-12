import { Injectable } from '@angular/core';
import { ChatItems, ChatItem } from '../models/chat-item';
import { HttpClient, HttpDownloadProgressEvent, HttpEvent, HttpEventType } from '@angular/common/http';
import { ChatMessageItem } from '../models/chat-message-item';
import { v4 as uuidv4 } from 'uuid';
import { BehaviorSubject, Observable } from 'rxjs';
import { ChatMessage } from '../components/chat-message/chat-message';
import { Chat } from '../components/chat/chat';


interface IChatsService {
  createChat(title: string): Promise<ChatItem>
  renameChat(chatUuid: string, newTitle: string): Promise<ChatItem>
  deleteChat(chatUuid: string): Promise<void>
  listChats(): Promise<ChatItem[]>
  addMessage(message: ChatMessageItem): Promise<ChatMessageItem>
  listMessages(chatUuid: string): Promise<ChatMessageItem[]>
  generateResponse(chatUuid: string): Observable<string>
}

@Injectable({
  providedIn: 'root'
})
export class ChatsService implements IChatsService {
  constructor(private http: HttpClient) { }

  createChat(title: string): Promise<ChatItem> {
    return new Promise((resolve, reject) => {
      this.http.post<ChatItem>('/api/chat/create', { title: title }).subscribe({
        next: (chat) => resolve(chat),
        error: (err) => reject(err)
      })
    });
  }

  renameChat(chatUuid: string, newTitle: string): Promise<ChatItem> {
    return new Promise((resolve, reject) => {
      this.http.post<ChatItem>('/api/chat/rename', { uuid: chatUuid, newTitle: newTitle }).subscribe({
        next: (chat) => resolve(chat),
        error: (err) => reject(err)
      })
    });
  }

  deleteChat(chatUuid: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.http.post<void>('/api/chat/delete', { uuid: chatUuid }).subscribe({
        next: () => resolve(),
        error: (err) => reject(err)
      })
    });
  }

  listChats(): Promise<ChatItem[]> {
    return new Promise((resolve, reject) => {
      this.http.get<ChatItems>('/api/chat/list').subscribe({
        next: (chats) => resolve(chats.chats),
        error: (err) => reject(err)
      })
    });
  }

  listMessages(chatUuid: string): Promise<ChatMessageItem[]> {
    return new Promise((resolve, reject) => {
      this.http.get<ChatMessageItem[]>(`/api/chat/list_messages/${chatUuid}`).subscribe({
        next: (messages) => resolve(messages),
        error: (err) => reject(err)
      })
    });
  }

  addMessage(message: ChatMessageItem): Promise<ChatMessageItem> {
    return new Promise((resolve, reject) => {
      this.http.post<ChatMessageItem>('/api/chat/add_message', message).subscribe({
        next: (msg) => resolve(msg),
        error: (err) => reject(err)
      })
    });
  }

  generateResponse(chatUuid: string): Observable<string> {
    return new Observable<string>(observer => {
      const eventSource = new EventSource(`/api/generate_response/${chatUuid}`);
      eventSource.onmessage = (event) => {
        if (event.data === "[DONE]") {
          observer.complete();
          eventSource.close();
          return;
        }
        observer.next(JSON.parse(event.data).text)
      };
      eventSource.onerror = (err) => {
        console.error("EventSource failed:", err);
        eventSource.close();
      };
    });
  }
}
