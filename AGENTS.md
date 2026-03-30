Не удаляй а наоборот пиши перед каждым смысловом блоком комментарий типа такого:
# Planner execution
                try:
                    validated_history = ensure_tool_response_sequence(self.message_histories["planner"])
                    self.logger.info(
                        "Validated planner history (%s messages):\n%s",
                        len(validated_history),
                        format_payload(validated_history),
                    )

                    planner_output = await create_plan(
                        planner_prompt,
                        message_history=validated_history,
                    )

                    planner_new_messages = planner_output.new_messages()
                    self.logger.info(
                        "Conversation handler history before append (%s messages):\n%s",
                        len(self.conversation_handler.conversation_history),
                        format_payload(self.conversation_handler.conversation_history),
                    )
                    self.conversation_handler.add_planner_message(planner_output)
                    self.logger.info(
                        "Conversation handler history after append (%s messages):\n%s",
                        len(self.conversation_handler.conversation_history),
                        format_payload(self.conversation_handler.conversation_history),
                    )

                    self.message_histories["planner"].extend(planner_new_messages)

                    plan = planner_output.output.plan
                    current_step = planner_output.output.next_step

                    self.logger.info("План от планировщика получен")
                    self.logger.info("Текущий шаг: %s", current_step)

                    if self.iteration_counter == 1:
                        self.logger.info("Полный план на первой итерации:\n%s", plan)
                except Exception:
                    self.logger.exception(
                        "Произошла ошибка планировщика на итерации %s",
                        self.iteration_counter,
                    )
                    raise


  # Browser Execution
  ...