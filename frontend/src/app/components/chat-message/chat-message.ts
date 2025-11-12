import { Component, Input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { ChatMessageItem } from '../../models/chat-message-item';

@Component({
  selector: 'app-chatmessage',
  imports: [MatIconModule],
  templateUrl: './chat-message.html',
  styleUrl: './chat-message.css'
})
export class ChatMessage {
  @Input() item: ChatMessageItem | null = null;

  get escapedContent(): string {
    if (!this.item || !this.item.content) return '';
    return this.escapeHtml(this.item.content);
  }

  private escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
}
