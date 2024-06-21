from typing import List, Optional
import uuid, math, os, textwrap
from utils import formatting_query_prompt, get_response_from_finetune_checkpoint, tokenizer

class Node:
    def __init__(self, content: str, parent: Optional['Node'] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.parent = parent
        self.children: List[Node] = []

    def add_child(self, child: 'Node'):
        self.children.append(child)

class LLMPlanner:
    def __init__(self, sys_prompt: str, tokenizer, goal: str):
        self.root = Node("Root")
        self.current_node = self.root
        self.sys_prompt = sys_prompt
        self.tokenizer = tokenizer
        self.message_history = []
        self.goal = goal

    def generate_plans(self, num_plans: int = 3) -> List[str]:
        plans = []
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts and len(plans) < num_plans:
            prompt = f"""
    Given the current goal: {self.goal}
    And the current path: {' -> '.join(self.get_current_path())}

    Generate {num_plans - len(plans)} possible next actions or plans. Each plan should be a concise, actionable step towards the goal.

    Format your response as a numbered list:
    1. [First plan]
    2. [Second plan]
    3. [Third plan]
    ...
    """
            self.message_history.append({"role": "user", "content": prompt})
            format_prompt = formatting_query_prompt(self.message_history, self.sys_prompt, self.tokenizer)
            response = get_response_from_finetune_checkpoint(format_prompt=format_prompt, do_print=False)
            
            # Parse the response to extract individual plans
            for line in response.strip().split('\n'):
                if line.strip().startswith(tuple(str(i) for i in range(1, num_plans + 1))):
                    plan = line.split('.', 1)[1].strip()
                    if plan not in plans:  # Avoid duplicates
                        plans.append(plan)
                    if len(plans) == num_plans:
                        break  # We have enough plans, exit the loop

            attempts += 1

        return plans


    def add_plans(self, plans: List[str]):
        for plan in plans:
            new_node = Node(plan, parent=self.current_node)
            self.current_node.add_child(new_node)

    def select_plan(self, plan_index: int):
        if 0 <= plan_index < len(self.current_node.children):
            self.current_node = self.current_node.children[plan_index]
            # Add the selected plan to the message history
            self.message_history.append({"role": "assistant", "content": f"Selected plan: {self.current_node.content}"})
        else:
            raise ValueError("Invalid plan index")

    def edit_plan(self, plan_index: int, new_content: str):
        if 0 <= plan_index < len(self.current_node.children):
            self.current_node.children[plan_index].content = new_content
            # Add the edited plan to the message history
            self.message_history.append({"role": "user", "content": f"Edited plan {plan_index} to: {new_content}"})
        else:
            raise ValueError("Invalid plan index")

    def display_current_plans(self):
        for i, child in enumerate(self.current_node.children):
            print(f"{i}: {child.content}")

    def get_current_path(self) -> List[str]:
        path = []
        node = self.current_node
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))
    
    def store_plans(self, file_path = ".plan_info/aplan.json") -> dict:
        """
        Store information into a dictionary object
        - Human input, edit, choice
        - Planning Path 
        - Target
        """
        stored_info = {
            "target": self.goal,
            "planning_path": self.get_current_path(),
            "current_plans": [child.content for child in self.current_node.children],
            "interaction_history": []
        }

        # Process message history to extract human interactions
        for message in self.message_history:
            if message["role"] == "user":
                if message["content"].startswith("Edited plan"):
                    stored_info["interaction_history"].append({
                        "type": "edit",
                        "content": message["content"]
                    })
                elif "Selected plan:" in message["content"]:
                    stored_info["interaction_history"].append({
                        "type": "choice",
                        "content": message["content"]
                    })
                else:
                    stored_info["interaction_history"].append({
                        "type": "input",
                        "content": message["content"]
                    })

        return stored_info

    
    def display_plans_in_columns(self, num_columns: int = 3):
        plans = self.current_node.children
        num_plans = len(plans)
        rows = math.ceil(num_plans / num_columns)

        column_width = 40  # Increased width to accommodate more text
        wrapper = textwrap.TextWrapper(width=column_width - 4, subsequent_indent='  ')

        print("┌" + "─" * (column_width * num_columns + num_columns - 1) + "┐")
        
        for row in range(rows):
            row_content = []
            max_lines = 1
            
            for col in range(num_columns):
                index = row * num_columns + col
                if index < num_plans:
                    plan = plans[index]
                    wrapped_content = wrapper.wrap(f"{index}: {plan.content}")
                    max_lines = max(max_lines, len(wrapped_content))
                    row_content.append(wrapped_content)
                else:
                    row_content.append([])
            
            for line in range(max_lines):
                line_content = []
                for col in range(num_columns):
                    if line < len(row_content[col]):
                        line_content.append(f"│{row_content[col][line]:<{column_width}}")
                    else:
                        line_content.append(f"│{' ' * column_width}")
                print("".join(line_content) + "│")
            
            if row < rows - 1:
                print("├" + ("─" * column_width + "┼") * (num_columns - 1) + "─" * column_width + "┤")

            print("└" + "─" * (column_width * num_columns + num_columns - 1) + "┘")

# Example usage:
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main(tokenizer=tokenizer):
    sys_prompt = "You are an AI assistant helping with planning and goal achievement."
    tokenizer = tokenizer
    goal = input("Enter your goal: ")
    
    planner = LLMPlanner(sys_prompt=sys_prompt, tokenizer=tokenizer, goal=goal)
    
    # Generate initial plans
    initial_plans = planner.generate_plans()
    planner.add_plans(initial_plans)

    while True:
        clear_screen()
        print("\n" + "=" * 80)
        print("🌳 LLM Planner 🌳".center(80))
        print("=" * 80 + "\n")

        print(f"Goal: {planner.goal}")
        path = " -> ".join(planner.get_current_path())
        print(f"Current path: {path}")
        print("\n" + "~" * 80 + "\n")

        print("Current plans:")
        planner.display_plans_in_columns()
        
        print("\n" + "~" * 80 + "\n")
        print("Actions:")
        print("  [S] Select a plan")
        print("  [E] Edit a plan")
        print("  [G] Generate more plans")
        print("  [Q] Quit")
        print("\n" + "~" * 80)
        
        action = input("\nEnter your choice: ").lower()
        
        if action == 'q':
            break
        elif action == 's':
            plan_index = int(input("Enter the index of the plan to select: "))
            planner.select_plan(plan_index)
            # Generate new plans for the next stage
            new_plans = planner.generate_plans()
            planner.add_plans(new_plans)
            print("Generated plans for the next stage:")
            planner.display_plans_in_columns()
            input("Press Enter to continue...")
        elif action == 'e':
            plan_index = int(input("Enter the index of the plan to edit: "))
            new_content = input("Enter the new content for the plan: ")
            planner.edit_plan(plan_index, new_content)
        elif action == 'g':
            new_plans = planner.generate_plans()
            planner.add_plans(new_plans)
            print("Generated additional plans:")
            planner.display_plans_in_columns()
            input("Press Enter to continue...")
        else:
            input("Invalid action. Press Enter to continue...")

    clear_screen()
    print("\n" + "=" * 80)
    print("Thank you for using LLM Planner!".center(80))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()