import { Component, Input } from '@angular/core';
import { JsonPipe } from '@angular/common';
import { JsonViewModule } from 'nxt-json-view'
import { ChatMessage } from '../chat-message/chat-message';
import { ChatMessageItem, ChatMessageItemFactory } from '../../models/chat-message-item';
import { MatTabsModule } from '@angular/material/tabs';

@Component({
  selector: 'app-agent-run',
  imports: [ChatMessage, JsonViewModule, MatTabsModule],
  templateUrl: './agent-run.html',
  styleUrl: './agent-run.css',
})
export class AgentRun {
  @Input() agentRun: any = null;
}
