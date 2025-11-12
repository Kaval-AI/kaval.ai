import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ChatItem, ChatItems } from '../models/chat-item';
import { ChatMessageItem } from '../models/chat-message-item';


interface ChatMessagesResponse {
  chat_id: string
  chat_messages: ChatMessageItem[]
}

interface ChatAgentRunsReponse {
  chat_id: string
  agent_runs: any[]
}


@Injectable({
  providedIn: 'root'
})
export class ChatsService {
  constructor(private http: HttpClient) { }

  listChats(): Promise<ChatItem[]> {
    return new Promise((resolve, reject) => {
      this.http.get<ChatItems>('/api/chat/list').subscribe({
        next: (chats) => resolve(chats.chats),
        error: (err) => reject(err)
      })
    });
  }

  listMessages(chatId: string): Promise<ChatMessageItem[]> {
    return new Promise((resolve, reject) => {
      this.http.get<ChatMessagesResponse>(`/api/chat/messages/${chatId}`).subscribe({
        next: (response) => resolve(response.chat_messages),
        error: (err) => reject(err)
      })
    });
  }

  listAgentRuns(chatId: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
      this.http.get<ChatAgentRunsReponse>(`/api/chat/agent_runs/${chatId}`).subscribe({
        next: (response) => resolve(response.agent_runs),
        error: (err) => reject(err)
      })
    });
  }

  getInputSchema(): Promise<any> {
    return new Promise((resolve, reject) => {
      this.http.get<any>(`/api/input_schema`).subscribe({
        next: (response) => resolve(response),
        error: (err) => reject(err)
      })
    });
  }

  runAgent(inputData: any): Promise<any> {
    return new Promise((resolve, reject) => {
      this.http.post<any>(`/api/agent/run`, inputData).subscribe({
        next: (response) => resolve(response),
        error: (err) => reject(err)
      })
    });
  }

    // createChat(title: string): Promise<ChatItem> {
  //   return new Promise((resolve, reject) => {
  //     this.http.post<ChatItem>('/api/chat/create', { title: title }).subscribe({
  //       next: (chat) => resolve(chat),
  //       error: (err) => reject(err)
  //     })
  //   });
  // }

  // addMessage(message: ChatMessageItem): Promise<ChatMessageItem> {
  //   return new Promise((resolve, reject) => {
  //     this.http.post<ChatMessageItem>('/api/chat/add_message', message).subscribe({
  //       next: (msg) => resolve(msg),
  //       error: (err) => reject(err)
  //     })
  //   });
  // }

  // generateResponse(chatUuid: string): Observable<string> {
  //   return new Observable<string>(observer => {
  //     const eventSource = new EventSource(`/api/generate_response/${chatUuid}`);
  //     eventSource.onmessage = (event) => {
  //       if (event.data === "[DONE]") {
  //         observer.complete();
  //         eventSource.close();
  //         return;
  //       }
  //       observer.next(JSON.parse(event.data).text)
  //     };
  //     eventSource.onerror = (err) => {
  //       console.error("EventSource failed:", err);
  //       eventSource.close();
  //     };
  //   });
  // }
}
